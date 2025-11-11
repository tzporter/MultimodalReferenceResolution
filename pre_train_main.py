
from __future__ import print_function

import warnings
# sys.path.extend(['../'])
import warnings
from sequential_parser import get_parser
import yaml
warnings.filterwarnings('ignore')
import datetime

import torch
from torch import optim
from torch import nn
from torch.utils.data import random_split
from feeders.gestures_feeder import CABBFeeder as FeederGestures
from utils.utils import load_yaml_to_dict
from callbacks.online_evaluation import OnlineEvalCallback
from collections import defaultdict

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from model.skeleton_speech_models import GSSModel
from utils.augmentation_utils import Augmenter2D
from model.losses import (
    NTXent,
    NTXentMM,
    VicReg,
    loss_2d_weighted,
    loss_velocity,
    loss_bone,
    WassersteinDistanceLoss
)


warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

class SpeechSkeletonModel(L.LightningModule):
    def __init__(self, arg):
        super().__init__()
        # build data loader
        self.arg = arg
        self.modalities = arg.model_args['modalities']
        self.skeleton_backbone = arg.model_args['skeleton_backbone']
        # TODO:
        # initialize models and losses based on opt and/or configs: unimodal, multimodal, contrastive loss
        self.model = GSSModel(**arg.model_args)
        self.freeze_bertje = arg.model_args['freeze_bertje']
        self.hidden_dim = arg.model_args['hidden_dim']
        self.bertje_dim = arg.model_args['bertje_dim']
        self.loss_types = arg.model_args['loss_types']

        assert not (('contrastive' in self.loss_types) and ('vicreg' in self.loss_types)), \
            "'contrastive' and 'vicreg' cannot both be in self.loss_types."

        assert not (('mm_contrastive' in self.loss_types) and ('mm_vicreg' in self.loss_types)), \
            "'mm_contrastive' and 'mm_vicreg' cannot both be in self.loss_types."

        if 'contrastive' in self.loss_types:
            self.contrastive_criterion = NTXent(batch_size=arg.batch_size, n_views=2, temperature=arg.temp)
        if 'matching' in self.loss_types:
            assert 'contrastive' not in self.loss_types, "Matching loss cannot be used with contrastive loss."
            assert 'masked_reconstruction' not in self.loss_types, "Matching loss cannot be used with masked rec."
            self.matching_criterion = nn.BCEWithLogitsLoss()
        if 'vicreg' in self.loss_types:
            self.vicreg_criterion = VicReg(
                embedding_size=self.hidden_dim,
                ssl_batch_size=arg.batch_size,
                sim_coeff=vars(arg).get('sim_coeff', 25),
                std_coeff=vars(arg).get('std_coeff', 25),
                cov_coeff=vars(arg).get('cov_coeff', 1)
            )
        if 'mm_contrastive' in self.loss_types and 'speech' in self.modalities:
            self.mm_speech_criterion = NTXentMM(batch_size=arg.batch_size, temperature=arg.temp)
        if 'mm_vicreg' in self.loss_types and 'speech' in self.modalities:
            self.mm_speech_criterion = VicReg(
                embedding_size=self.hidden_dim,
                ssl_batch_size=arg.batch_size,
                sim_coeff=vars(arg).get('sim_coeff', 25),
                std_coeff=vars(arg).get('std_coeff', 25),
                cov_coeff=vars(arg).get('cov_coeff', 1)
            )
        if 'semantic' in self.modalities and 'mm_contrastive' in self.loss_types:
            self.mm_semantic_criterion = NTXentMM(batch_size=arg.batch_size, temperature=arg.temp)
        if 'mm_vicreg' in self.loss_types and 'semantic' in self.modalities:
            self.mm_semantic_criterion = VicReg(
                embedding_size=self.hidden_dim,
                ssl_batch_size=arg.batch_size,
                sim_coeff=vars(arg).get('sim_coeff', 25),
                std_coeff=vars(arg).get('std_coeff', 25),
                cov_coeff=vars(arg).get('cov_coeff', 1)
            )
        if 'masked_reconstruction' in self.loss_types:
            self.augmenter2D = Augmenter2D(mask_ratio=arg.mask_ratio, mask_T_ratio=arg.mask_T_ratio)
        if 'wasserstein' in self.loss_types:
            self.wasserstein_criterion = WassersteinDistanceLoss()

        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=arg.learning_rate,
            weight_decay=arg.weight_decay
        )
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)
        # TODO sync batch norm
        # model = apex.parallel.convert_syncbn_model(model)

    def _compute_loss_2d(self, predicted_pos, batch_gt, conf, phase):
        losses = defaultdict()
        loss_2d_proj = loss_2d_weighted(predicted_pos, batch_gt, conf)
        loss_velocity_2d = loss_velocity(predicted_pos, batch_gt)
        loss_bone_2d = loss_bone(predicted_pos, batch_gt)
        losses[f'{phase}/joints_loss'] = loss_2d_proj
        losses[f'{phase}/velocity_loss'] = loss_velocity_2d
        losses[f'{phase}/bone_loss'] = loss_bone_2d
        total_loss = (loss_2d_proj + loss_velocity_2d + loss_bone_2d) / 3
        losses[f'{phase}/total_loss'] = total_loss
        return total_loss, losses

    def process_batch(self, batch):
        if "skeleton" in self.modalities:
            orig_skeletons = batch["skeleton"]["orig"] if "orig" in batch["skeleton"] else None
            skeletons_1 = batch["skeleton"]["view1"] if "view1" in batch["skeleton"] else None
            skeletons_2 = batch["skeleton"]["view2"] if "view2" in batch["skeleton"] else None
            batch_gt = batch['skeleton']['orig'] if 'orig' in batch['skeleton'] else None
            conf = orig_skeletons[:, :, :, 2:]
        else:
            skeletons_1 = None
            skeletons_2 = None
            orig_skeletons = None
            batch_gt = None
        if "speech" in self.modalities:
            speech_1 = batch["speech"]["view1"] if "view1" in batch["speech"] else None
            orig_speech = batch["speech"]["orig"] if "orig" in batch["speech"] else None
            speech_1 = batch["speech"]["view1"] if "view1" in batch["speech"] else None
            speech_2 = batch["speech"]["view2"] if "view2" in batch["speech"] else None
            speech_lengths = batch["speech"]["lengths"] if "lengths" in batch["speech"] else None
        else:
            orig_speech = None
            speech_1 = None
            speech_2 = None
            speech_lengths = None

        if "semantic" in self.modalities:
            utterances = batch["utterance"]
        else:
            utterances = None

        label = batch["label"]
        return {
            "orig_skeletons": orig_skeletons,
            "skeletons_1": skeletons_1,
            "skeletons_2": skeletons_2,
            "orig_speech": orig_speech,
            "speech_1": speech_1,
            "speech_2": speech_2,
            "labels": label,
            "speech_lengths": speech_lengths,
            "utterance": utterances,
            "batch_gt": batch_gt,
            "conf": conf
        }

    def _shared_step(self, batch, prefix="train"):
        processed_batch = self.process_batch(batch)
        processed_batch["orig_skeletons"] = processed_batch["orig_skeletons"].float() if processed_batch["orig_skeletons"] is not None else None
        processed_batch["skeletons_1"] = processed_batch["skeletons_1"].float() if processed_batch["skeletons_1"] is not None else None
        processed_batch["skeletons_2"] = processed_batch["skeletons_2"].float() if processed_batch["skeletons_2"] is not None else None

        if processed_batch["skeletons_1"] is None and processed_batch["skeletons_2"] is None:
            skeletons = processed_batch["orig_skeletons"]
        elif processed_batch["skeletons_1"] is not None and processed_batch["skeletons_2"] is None:
            skeletons = torch.cat([processed_batch["orig_skeletons"], processed_batch["skeletons_1"]], dim=0)
        else:
            skeletons = torch.cat([processed_batch["skeletons_1"], processed_batch["skeletons_2"]], dim=0)
        if "masked_reconstruction" in self.loss_types and not "contrastive" in self.loss_types:
            # augment skeleton
            skeletons = self.augmenter2D.augment2D(processed_batch["orig_skeletons"], mask=True, noise=True)
        if "contrastive" in self.loss_types and "masked_reconstruction" in self.loss_types:
            # assert that there are two views of the skeletons and the original skeletons
            assert processed_batch["skeletons_1"] is not None and processed_batch["skeletons_2"] is not None
            assert processed_batch["orig_skeletons"] is not None 
            skeletons = self.augmenter2D.augment2D(processed_batch["orig_skeletons"], mask=True, noise=True)
            skeletons = torch.cat([skeletons, processed_batch["skeletons_1"], processed_batch["skeletons_2"]], dim=0)
        elif "matching" in self.loss_types:
            skeletons = processed_batch["orig_skeletons"]
        if "speech" in self.modalities:
            speech_lengths = processed_batch["speech_lengths"]
            processed_batch["orig_speech"] = processed_batch["orig_speech"].float() if processed_batch["orig_speech"] is not None else None
            processed_batch["speech_1"] = processed_batch["speech_1"].float() if processed_batch["speech_1"] is not None else None
            processed_batch["speech_2"] = processed_batch["speech_2"].float() if processed_batch["speech_2"] is not None else None

            if processed_batch["speech_1"] is None and processed_batch["speech_2"] is None:
                speech = processed_batch["orig_speech"]
            elif processed_batch["speech_1"] is not None and processed_batch["speech_2"] is None:
                speech = torch.cat([processed_batch["orig_speech"], processed_batch["speech_1"]], dim=0)
            else:
                speech = torch.cat([processed_batch["speech_1"], processed_batch["speech_2"]], dim=0)
        else:
            speech = None
            speech_lengths = None

        loss = 0
        neg_similarity = 0
        pos_similarity = 0
        # forward pass only if constrastive loss is used
        if (
            "contrastive" in self.loss_types or
            "mm_contrastive" in self.loss_types or
            "vicreg" in self.loss_types or
            "mm_vicreg" in self.loss_types or
            "masked_reconstruction" in self.loss_types or
            "matching" in self.loss_types
        ):
            if "matching" in self.loss_types:
                # shuffle utterances and create binary labels:
                # 0 for non-matching (shuffled), 1 for matching (non-shuffled)
                assert len(skeletons) == len(processed_batch["utterance"]), \
                    "Skeletons and utterances must have the same number of samples for matching loss."
                if len(processed_batch["utterance"]) % 2 != 0:
                    processed_batch["utterance"] = processed_batch["utterance"][:-1]
                    skeletons = skeletons[:-1]

                non_shuffled_utterances = processed_batch["utterance"][:len(processed_batch["utterance"]) // 2]
                shuffled_utterances = processed_batch["utterance"][len(processed_batch["utterance"]) // 2:]
                indices = torch.randperm(len(shuffled_utterances))
                shuffled_utterances = [shuffled_utterances[i] for i in indices]

                processed_batch["utterance"] = non_shuffled_utterances + shuffled_utterances
                binary_labels = torch.zeros(len(processed_batch["utterance"]), dtype=torch.float32)
                binary_labels[:len(non_shuffled_utterances)] = 1.
                binary_labels = binary_labels.to(self.device)

            representations = True if (
                "contrastive" in self.loss_types or
                "mm_contrastive" in self.loss_types or
                "vicreg" in self.loss_types or
                "mm_vicreg" in self.loss_types or
                "matching" in self.loss_types
            ) else False
            reconstruction = True if "masked_reconstruction" in self.loss_types else False
            features = self.model(
                skeleton=skeletons,
                speech_waveform=speech,
                utterances=processed_batch["utterance"],
                speech_lengths=speech_lengths,
                vicreg=True if "vicreg" in self.loss_types or "mm_vicreg" in self.loss_types else False,
                representations=representations,
                reconstruction=reconstruction,
            )
            skeleton_features = features.get("skeleton_features", None)
            speech_features = features.get("speech_features", None)
            semantic_features = features.get("semantic_features", None)
            semantic_local_features = features.get("semantic_local_features", None)
            skeleton_local_features = features.get("skeleton_local_features", None)
            skeleton_predictions = features.get("skeleton_predictions", None)

            # Contrastive losses -------------------------------------------------- #
            if "masked_reconstruction" in self.loss_types:
                # check if the skeleton predictions are twice the size of the original skeletons                    
                # the masked skeleton is always the first part of the batch
                skeleton_predictions = skeleton_predictions[:int(processed_batch["batch_gt"].shape[0]), ...] 
                total_loss, masked_losses_d = self._compute_loss_2d(
                    skeleton_predictions,
                    processed_batch["batch_gt"],
                    processed_batch["conf"],
                    phase=prefix
                )
                loss += total_loss
                for loss_name in masked_losses_d.keys():
                    self.log(loss_name, masked_losses_d[loss_name])
            # Unimodal gesture
            if "contrastive" in self.loss_types:
                if "masked_reconstruction" in self.loss_types:
                    # the skeleton features are the second and third part of the batch
                    skeleton_features = skeleton_features[int(skeleton_features.shape[0] // 3):, ...]
                skeleton_loss, skeleton_pos, skeleton_neg = self.contrastive_criterion(skeleton_features)
                self.log(f'{prefix}/skeleton_loss', skeleton_loss)
                self.log(f'{prefix}/skeleton_pos', skeleton_pos)
                self.log(f'{prefix}/skeleton_neg', skeleton_neg)
                loss += skeleton_loss
                neg_similarity += skeleton_neg
                pos_similarity += skeleton_pos
            if 'matching' in self.loss_types:
                matching_loss = self.matching_criterion(features['matching_predictions'].squeeze(), binary_labels)
                loss += matching_loss
                self.log(f'{prefix}/matching_loss', matching_loss)
            # Multimodal gesture-speech
            if 'speech' in self.modalities and 'mm_contrastive' in self.loss_types:
                mm_loss, mm_pos, mm_neg = self.mm_speech_criterion(
                    speech_features,
                    skeleton_features[:int(skeleton_features.shape[0] // 2), :]
                )
                loss += mm_loss
                neg_similarity += mm_neg
                pos_similarity += mm_pos
                self.log(f'{prefix}/mm_loss', mm_loss)
                self.log(f'{prefix}/mm_pos', mm_pos)
                self.log(f'{prefix}/mm_neg', mm_neg)
            # Multimodal gesture-semantic
            if 'semantic' in self.modalities and 'mm_contrastive' in self.loss_types:
                semantic_mm_loss, semantic_mm_pos, semantic_mm_neg = self.mm_semantic_criterion(
                    semantic_features,
                    skeleton_features[:int(skeleton_features.shape[0] // 2), :])
                loss += semantic_mm_loss
                neg_similarity += semantic_mm_neg  
                pos_similarity += semantic_mm_pos
                self.log(f'{prefix}/semantic_mm_loss', semantic_mm_loss)
                self.log(f'{prefix}/semantic_mm_pos', semantic_mm_pos)
                self.log(f'{prefix}/semantic_mm_neg', semantic_mm_neg)

            # VicReg losses ---------------------------------------------------------------- #
            # Unimodal gesture
            if 'vicreg' in self.loss_types:
                (
                    skeleton_loss,
                    skeleton_repr_loss,
                    skeleton_std_loss,
                    skeleton_cov_loss
                ) = self.vicreg_criterion(
                    skeleton_features[:int(skeleton_features.shape[0] // 2), ...],
                    skeleton_features[int(skeleton_features.shape[0] // 2):, ...]
                )
                loss += skeleton_loss
                self.log(f'{prefix}/skeleton_vicreg_loss', skeleton_loss)
                self.log(f'{prefix}/skeleton_vicreg_repr_loss', skeleton_repr_loss)
                self.log(f'{prefix}/skeleton_vicreg_std_loss', skeleton_std_loss)
                self.log(f'{prefix}/skeleton_vicreg_cov_loss', skeleton_cov_loss)
            # Multimodal gesture-speech
            if 'speech' in self.modalities and 'mm_vicreg' in self.loss_types:
                (
                    speech_mm_loss,
                    speech_mm_repr_loss,
                    speech_mm_std_loss,
                    speech_mm_cov_loss
                ) = self.mm_speech_criterion(
                    speech_features,
                    skeleton_features[:int(skeleton_features.shape[0] // 2), ...]
                )
                loss += speech_mm_loss
                self.log(f'{prefix}/speech_vicreg_loss', speech_mm_loss)
                self.log(f'{prefix}/speech_vicreg_repr_loss', speech_mm_repr_loss)
                self.log(f'{prefix}/speech_vicreg_std_loss', speech_mm_std_loss)
                self.log(f'{prefix}/speech_vicreg_cov_loss', speech_mm_cov_loss)
            # Multimodal gesture-semantic
            if 'semantic' in self.modalities and 'mm_vicreg' in self.loss_types:
                (
                    semantic_mm_loss,
                    semantic_mm_repr_loss,
                    semantic_mm_std_loss,
                    semantic_mm_cov_loss
                ) = self.mm_semantic_criterion(
                    semantic_features,
                    skeleton_features[:int(skeleton_features.shape[0] // 2), ...]
                )
                loss += semantic_mm_loss
                self.log(f'{prefix}/semantic_vicreg_loss', semantic_mm_loss)
                self.log(f'{prefix}/semantic_vicreg_repr_loss', semantic_mm_repr_loss)
                self.log(f'{prefix}/semantic_vicreg_std_loss', semantic_mm_std_loss)
                self.log(f'{prefix}/semantic_vicreg_cov_loss', semantic_mm_cov_loss)

            # Wasserstein distance
            if (
                'wasserstein' in self.loss_types and
                semantic_local_features is not None and
                skeleton_local_features is not None
            ):
                wasserstein_loss = self.wasserstein_criterion(
                    skeleton_local_features[:int(skeleton_local_features.shape[0] // 2), ...],
                    semantic_local_features
                )
                loss += wasserstein_loss
                self.log(f'{prefix}/wasserstein_loss_skeleton_semantic', wasserstein_loss)

        loss = loss / len(self.loss_types)
        self.log(f'{prefix}/combined_loss', loss)
        if (
            "contrastive" in self.loss_types or
            "mm_contrastive" in self.loss_types
        ):
            self.log(f'{prefix}/combined_neg', neg_similarity/len(self.loss_types))
            self.log(f'{prefix}/combined_pos', pos_similarity/len(self.loss_types))

        return loss

    def training_step(self, batch, batch_idx):
        # Prepare data
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        skeleton_1, skeleton_2, speech_1, speech_2, labels = self.process_batch(batch)

    def on_train_epoch_end(self, outputs=None) -> None:
        # print loss, time, and other info
        print('end of epoch')
        print('Current epoch: {}'.format(self.current_epoch))
        print('Current lr: {}'.format(self.optimizer.param_groups[0]['lr']))
        print('current loss is {}'.format(self.trainer.callback_metrics['train/combined_loss']))

    def on_test_epoch_end(self, outputs=None) -> None:
        pass

    def configure_optimizers(self):
        constant_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: 1.
        )

        return (
            [self.optimizer],
            [
                {
                    "scheduler": constant_lr_scheduler,
                    "interval": "epoch",
                    "monitor": "val/combined_loss"
                }
            ]
        )

def main(phase='training'):
    L.seed_everything(42)

    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
                if k not in key:
                    print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    modalities = arg.model_args["modalities"]
    arg.feeder_args["modalities"] = modalities
    arg.model_args["maxlen"] = arg.feeder_args["window_size"]

    skeleton_augmentations = load_yaml_to_dict(arg.skeleton_augmentations_path)

    # Splitting the dataset
    arg.feeder_args['skeleton_augmentations'] = skeleton_augmentations
    
    # By default we use CABB-XL for pretraining.
    dataset = FeederGestures(**arg.feeder_args)
    poses = dataset.poses
    mirrored_poses = dataset.mirrored_poses
    audio_dict = dataset.audio_dict
    # Assuming you have a dataset object
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size

    # Splitting the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=arg.batch_size, shuffle=(train_sampler is None),
        num_workers=arg.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=arg.batch_size,
        shuffle=False,
        num_workers=arg.num_workers,
        drop_last=True if arg.loss_function == "NTXentMM" else False
    )

    models_directory = 'workdir/'
    model = SpeechSkeletonModel(arg)

    # convert the list of modalities to a string
    logger_name = arg.Experiment_name.format("_".join(modalities), arg.learning_rate, arg.batch_size, arg.temp)
    experiment_id = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
    tb_logger = TensorBoardLogger(models_directory, name=logger_name, version=experiment_id)

    loggers = [tb_logger]

    # if arg.wandb_entity != "none":
    #     experiment_info = vars(arg)
    #     # project = "CABB_pretraining" if phase == 'training' else "CABB_testing"
    #     wandb_logger = WandbLogger(
    #         config=experiment_info,
    #         entity=arg.wandb_entity,
    #         project="first_test",
    #         name=experiment_id,
    #         id=experiment_id
    #     )
    #     loggers.append(wandb_logger)       
   
    callbacks = [
        # Monitor learning rate during training
        LearningRateMonitor(logging_interval='epoch'),
        # Save top 10 models based on "val/correlation" score, checked every epoch
        ModelCheckpoint(
            filename="{epoch}-{val/correlation:.2f}",
            monitor="val/correlation",
            save_top_k=5,
            every_n_epochs=1,
            mode="max"
        ),
        # Save top 10 models based on "val/difference" score, checked every epoch
        ModelCheckpoint(
            filename="{epoch}-{val/difference:.2f}",
            monitor="val/difference",
            save_top_k=5,
            every_n_epochs=1,
            mode="max"
        ),
        # Online evaluation callback for retrieval and similarity tasks. If you want to use this one, please use the correct retrieval and similarity metrics. For now we only use the similarity metric.
        OnlineEvalCallback(
            poses,
            mirrored_poses,
            audio_dict=audio_dict,
            every_n_epochs=[4, 1],
            tasks=["similarity"],
            skeleton_backbone=arg.model_args['skeleton_backbone'],
            modalities=modalities,
        ),
        ModelSummary(max_depth=2)
    ]

    # TODO: use all parameters either from configs or from opt
    if torch.cuda.is_available():
        # TODO: make mixed-precision optional from configs/opt to avoid errors (bf16)
        trainer = L.Trainer(
            # gradient_clip_val=0.25,
            max_epochs=arg.num_epoch,
            logger=loggers,
            accelerator="gpu",
            devices=arg.device,
            num_nodes=1,
            # accumulate_grad_batches=arg.accumulate_grad_batches,
            callbacks=callbacks,
            strategy="ddp_find_unused_parameters_true",
            enable_progress_bar=True,
            # precision="bf16",
            num_sanity_val_steps=2,
            default_root_dir=models_directory+arg.Experiment_name
            # show the progress bar
            )
    else:
        trainer = L.Trainer(
            gradient_clip_val=0.25,
            max_epochs=arg.num_epoch,
            logger=loggers,
            callbacks=[LearningRateMonitor(logging_interval='epoch')]
            )
    if phase == 'training':
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
