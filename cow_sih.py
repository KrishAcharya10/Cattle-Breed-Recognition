import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, EfficientNetV2B0, ResNet152V2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

AUTOTUNE = tf.data.AUTOTUNE

# ✅ FIX 1: Augmentation only for training (no validation/test augmentation)
def create_advanced_augmentation():
    """Advanced data augmentation - ONLY for training"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="advanced_augmentation")

# ✅ FIX 2: Proper preprocessing functions for each model
def preprocess_for_mobilenet(x):
    """MobileNetV2 expects [-1, 1] range"""
    return tf.keras.applications.mobilenet_v2.preprocess_input(x)

def preprocess_for_efficientnet(x):
    """EfficientNetV2 expects [0, 255] range (has built-in rescaling)"""
    return x  # EfficientNetV2 has built-in preprocessing

def preprocess_for_resnet(x):
    """ResNet expects specific preprocessing"""
    return tf.keras.applications.resnet_v2.preprocess_input(x)

def build_datasets_enhanced(data_dir, img_size=(224, 224), batch_size=32, use_class_weights=True):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # ✅ FIX 3: Load datasets without any preprocessing first
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size, shuffle=True, seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size, shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch_size, shuffle=False
    )

    class_names = train_ds.class_names

    # Compute class weights
    if use_class_weights:
        train_labels = []
        for _, labels in train_ds:
            train_labels.extend(labels.numpy())
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weight_dict = dict(zip(np.unique(train_labels), class_weights))
    else:
        class_weight_dict = None

    # Cache and prefetch
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names, class_weight_dict

def apply_augmentation_and_preprocessing(ds, preprocess_fn, augmentation_layer=None, training=False):
    """Apply preprocessing and optionally augmentation"""
    def process(x, y):
        if training and augmentation_layer is not None:
            x = augmentation_layer(x, training=True)
        x = preprocess_fn(x)
        return x, y
    
    return ds.map(process, num_parallel_calls=AUTOTUNE)

def build_model_mobilenet(num_classes, img_size=(224, 224), dropout_rate=0.5):
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(img_size[0], img_size[1], 3))
    base.trainable = False

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs), base

def build_model_efficientnet(num_classes, img_size=(224, 224), dropout_rate=0.5):
    base = EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(img_size[0], img_size[1], 3))
    base.trainable = False

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs), base

def build_model_resnet(num_classes, img_size=(224, 224), dropout_rate=0.5):
    base = ResNet152V2(include_top=False, weights="imagenet", input_shape=(img_size[0], img_size[1], 3))
    base.trainable = False

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs), base

def create_advanced_callbacks(model_name):
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1, mode='max'),
        ModelCheckpoint(f"best_{model_name}.h5", monitor="val_accuracy", save_best_only=True, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]
    return callbacks

def focal_loss(gamma=2., alpha=0.25):
    """Focal loss for handling class imbalance"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        if len(y_true.shape) == 1:
            y_true = tf.cast(y_true, tf.int32)
            y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        else:
            y_true = tf.cast(y_true, tf.float32)

        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        ce = -tf.math.log(pt)
        focal_loss = alpha * tf.pow(1 - pt, gamma) * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

    return focal_loss_fixed

def train_ensemble(data_dir="data", epochs=20, fine_tune_epochs=10, batch_size=32, img_size=(224, 224)):
    # Load base datasets
    train_ds, val_ds, test_ds, class_names, class_weight_dict = build_datasets_enhanced(
        data_dir, img_size=img_size, batch_size=batch_size
    )
    num_classes = len(class_names)
    
    # Create augmentation layer once
    augmentation = create_advanced_augmentation()

    models = {}
    model_configs = {
        'mobilenet': {
            'builder': build_model_mobilenet,
            'preprocess': preprocess_for_mobilenet
        },
        'efficientnet': {
            'builder': build_model_efficientnet,
            'preprocess': preprocess_for_efficientnet
        },
        'resnet': {
            'builder': build_model_resnet,
            'preprocess': preprocess_for_resnet
        }
    }

    for model_name, config in model_configs.items():
        print(f"\n🚀 Training {model_name.upper()} model...")

        # ✅ FIX 4: Apply model-specific preprocessing
        preprocess_fn = config['preprocess']
        train_processed = apply_augmentation_and_preprocessing(
            train_ds, preprocess_fn, augmentation_layer=augmentation, training=True
        )
        val_processed = apply_augmentation_and_preprocessing(
            val_ds, preprocess_fn, augmentation_layer=None, training=False
        )
        test_processed = apply_augmentation_and_preprocessing(
            test_ds, preprocess_fn, augmentation_layer=None, training=False
        )

        model, base = config['builder'](num_classes, img_size)
        callbacks = create_advanced_callbacks(model_name)

        # Stage 1: Train head only
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=["accuracy"],
        )

        print(f"Stage 1: Training {model_name} head...")
        history1 = model.fit(
            train_processed,
            validation_data=val_processed,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        # Stage 2: Fine-tune last layers
        base.trainable = True
        for layer in base.layers[:-50]:  # Unfreeze more layers
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=["accuracy"],
        )

        print(f"Stage 2: Fine-tuning {model_name}...")
        history2 = model.fit(
            train_processed,
            validation_data=val_processed,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        test_loss, test_acc = model.evaluate(test_processed, verbose=0)
        print(f"✅ {model_name.upper()} Test Accuracy: {test_acc * 100:.2f}%")

        # ModelCheckpoint above already saved the best-val_accuracy weights to
        # best_{model_name}.h5; saving again here just duplicates a large file.
        models[model_name] = {'model': model, 'preprocess': preprocess_fn}

    # Create ensemble predictions
    print("\n🎯 Creating ensemble predictions...")
    ensemble_predictions = []
    test_labels = []

    for x_batch, y_batch in test_ds:
        batch_predictions = []
        for model_name, model_dict in models.items():
            # Apply correct preprocessing for each model
            x_processed = model_dict['preprocess'](x_batch)
            pred = model_dict['model'].predict(x_processed, verbose=0)
            batch_predictions.append(pred)

        ensemble_pred = np.mean(batch_predictions, axis=0)
        ensemble_predictions.append(ensemble_pred)
        test_labels.append(y_batch.numpy())

    ensemble_predictions = np.vstack(ensemble_predictions)
    test_labels = np.hstack(test_labels)
    ensemble_accuracy = np.mean(np.argmax(ensemble_predictions, axis=1) == test_labels)

    print(f"\n🎉 ENSEMBLE ACCURACY: {ensemble_accuracy * 100:.2f}%")

    # Save results
    np.save("ensemble_predictions.npy", ensemble_predictions)
    np.save("test_labels.npy", test_labels)
    with open("class_labels.pkl", "wb") as f:
        pickle.dump(class_names, f)

    print("\n📊 INDIVIDUAL MODEL PERFORMANCES:")
    for model_name, model_dict in models.items():
        test_processed = apply_augmentation_and_preprocessing(
            test_ds, model_dict['preprocess'], augmentation_layer=None, training=False
        )
        _, acc = model_dict['model'].evaluate(test_processed, verbose=0)
        print(f"   {model_name.upper()}: {acc * 100:.2f}%")

    return models, ensemble_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    args = parser.parse_args()

    models, ensemble_acc = train_ensemble(
        data_dir=args.data_dir,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size)
    )

    print(f"\n🎯 FINAL ENSEMBLE ACCURACY: {ensemble_acc * 100:.2f}%")
    print("Expected accuracy range: 85-95% based on research findings")
