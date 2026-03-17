"""
Configuration file for Fault Distance Detection CNN
All hyperparameters and paths in one place
"""

class Config:
    # ============ DATA PARAMETERS ============
    CSV_PATH = "data/oscillograms.csv"  # Path to your oscillogram data
    SEQ_LENGTH = 300                     # 100ms pre-fault + 200ms fault = 300 samples
    NUM_CHANNELS = 1                     # 1 for single signal (current/voltage)
                                         # Increase to 3-5 for multi-channel (U, I, P)
    TRAIN_SPLIT = 0.8                   # 80% train, 20% test
    
    # ============ MODEL ARCHITECTURE ============
    BASE_CHANNELS = 64                   # Base number of channels
                                         # 32 = faster, 128 = more accurate
    DROPOUT_RATE = 0.3                  # Dropout probability (0.2-0.5)
    USE_RESIDUAL = True                 # True = 1D-ResNet, False = simple CNN
    USE_SE_BLOCK = True                 # Squeeze-and-Excitation attention
    DEPTH = 3                            # Number of residual stages (2-4)
    
    # ============ TRAINING PARAMETERS ============
    BATCH_SIZE = 32                      # Batch size (16/32/64)
    EPOCHS = 100                         # Maximum epochs
    LEARNING_RATE = 1e-3                # Initial learning rate
    WEIGHT_DECAY = 1e-5                 # L2 regularization
    OPTIMIZER = "adam"                  # "adam" or "sgd"
    
    # ============ LEARNING RATE SCHEDULER ============
    LR_FACTOR = 0.5                     # Multiply LR by this factor
    LR_PATIENCE = 5                     # Reduce LR if val_loss not improving
    
    # ============ EARLY STOPPING ============
    EARLY_STOPPING_PATIENCE = 15        # Stop if val_loss not improving
    GRADIENT_CLIP = 1.0                 # Clip gradients to prevent explosion
    
    # ============ DEVICE & PRECISION ============
    DEVICE = "cuda"                     # "cuda" or "cpu"
    MIXED_PRECISION = False             # AMP (Automatic Mixed Precision) - experimental
    
    # ============ PATHS ============
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    LOGS_DIR = "logs"
    
    # ============ REGRESSION PARAMETERS ============
    TASK = "regression"                 # "regression" (predict distance) or "classification"
    NORMALIZE_DATA = True               # Standardize input signals
    
    # ============ AUGMENTATION (Optional) ============
    USE_AUGMENTATION = False            # Data augmentation
    AUGMENTATION_NOISE_STD = 0.01      # Gaussian noise std
    AUGMENTATION_SHIFT = 5              # Time shift in samples
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        assert cls.SEQ_LENGTH == 300, "Sequence length must be 300 (100ms + 200ms)"
        assert cls.TRAIN_SPLIT > 0 and cls.TRAIN_SPLIT < 1
        assert cls.BATCH_SIZE > 0
        assert cls.EPOCHS > 0
        assert cls.LEARNING_RATE > 0
        print("✅ Configuration validated successfully")
