import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from collections import defaultdict

# Local module imports
from src.config import CHAPTERS_JSON_PATH, SPEAKER_SEGMENTS_PATH, FPS, WINDOW_SIZE
from src.data_utils import load_raw_files, parse_csv_windows, RawExtractionDataset
from src.feature_extractor import extract_features
from src.classification import train_and_evaluate_classifiers, report_results

def main():
    """
    Main script to run the k-fold cross-validation experiment.
    """
    # --- Step 1: Load Configuration and Raw Data ---
    print("Loading configuration and raw data...")
    with open(CHAPTERS_JSON_PATH, "r") as f:
        ch_dict = json.load(f)
    all_chapters = sorted(ch_dict['train'] + ch_dict['test'])
    pose_data_map = {ch: data for ch, data in zip(all_chapters, load_raw_files(all_chapters, type='pose'))}
    audio_data_map = {ch: data for ch, data in zip(all_chapters, load_raw_files(all_chapters, type='audio'))}
    print(f"Loaded data for {len(all_chapters)} chapters.")

    # --- Step 2: Prepare for Cross-Validation ---
    segments_df = pd.read_csv(SPEAKER_SEGMENTS_PATH)
    all_categories = sorted(segments_df['category'].unique()) # We'll use all categories in each fold
    
    # K-Fold setup for chapters
    num_folds = 10
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    print(f"\nPreparing for {num_folds}-fold cross-validation, splitting by chapter.")

    all_folds_results = defaultdict(list)

    # --- Step 3: Run K-Fold Cross-Validation ---
    for i, (train_chapter_indices, val_chapter_indices) in enumerate(kf.split(all_chapters)):
        train_chapters = [all_chapters[i] for i in train_chapter_indices]
        val_chapters = [all_chapters[i] for i in val_chapter_indices]

        print("\n" + "="*50)
        print(f"FOLD {i+1}/{num_folds}: Validating on chapters {val_chapters}")
        print("="*50)

        # 3a. Load data for the current fold's chapters
        train_pose_data, train_audio_data = [pose_data_map[c] for c in train_chapters], [audio_data_map[c] for c in train_chapters]
        val_pose_data, val_audio_data = [pose_data_map[c] for c in val_chapters], [audio_data_map[c] for c in val_chapters]
        
        # 3b. Create window indices and labels for train/val sets
        print("Parsing CSV for train/val windows...")
        train_indices, train_labels = parse_csv_windows(segments_df, train_chapters, all_categories, FPS)
        val_indices, val_labels = parse_csv_windows(segments_df, val_chapters, all_categories, FPS)
        
        if not val_indices:
            print(f"Skipping fold {i+1} as there is no data for validation chapters {val_chapters}.")
            continue

        print(f"Defined {len(train_indices)} training windows and {len(val_indices)} validation windows.")

        # 3c. Create datasets for feature extraction
        train_ds_raw = RawExtractionDataset(train_pose_data, train_audio_data, train_indices, train_labels, WINDOW_SIZE)
        val_ds_raw = RawExtractionDataset(val_pose_data, val_audio_data, val_indices, val_labels, WINDOW_SIZE)

        # 3d. Extract features
        print("\n--- Phase 1: Feature Extraction ---", flush=True)
        X_train_full, y_train = extract_features(train_ds_raw)
        X_val_full, y_val = extract_features(val_ds_raw)

        # 3e. Train classifiers and get results
        print("\n--- Phase 2: Classification ---", flush=True)
        # This function now handles PCA, training, and evaluation for the fold
        classifier_results = train_and_evaluate_classifiers(X_train_full, y_train, X_val_full, y_val)

        # 3f. Report and store results for the fold
        fold_summary = report_results(classifier_results, y_train, y_val)
        for model_name, metrics in fold_summary.items():
            all_folds_results[model_name].append(metrics)

    # --- Step 4: Aggregate and Final Report ---
    print("\n\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Avg Val Accuracy':<20} {'Std Dev':<20}")
    print("-"*80)

    final_summary = {}
    for model_name, fold_data in all_folds_results.items():
        val_accuracies = [d['val_accuracy'] for d in fold_data]
        avg_acc = np.mean(val_accuracies)
        std_acc = np.std(val_accuracies)
        print(f"{model_name:<20} {avg_acc:.4f} ({avg_acc*100:5.2f}%)       {std_acc:.4f}")
        final_summary[model_name] = {
            'avg_val_accuracy': avg_acc,
            'std_dev_val_accuracy': std_acc,
            'fold_results': fold_data
        }

    with open('final_cv_results.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    print("\nSaved final cross-validation results to final_cv_results.json")

if __name__ == "__main__":
    main()