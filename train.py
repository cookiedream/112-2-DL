import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from dataloader import x_train, y_train, x_test, y_test, datagen
from model import *
import os
from tensorflow_addons.optimizers import AdamW
from rich.progress import Progress, BarColumn, TimeRemainingColumn, ProgressColumn, TextColumn, SpinnerColumn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from keras.callbacks import EarlyStopping, ModelCheckpoint
from contextlib import redirect_stdout


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# hyperparameters
batch_size = 128
shuffle = True
epochs_num = 2000
patience = 200  # 設定 early stop 的等待次數
lr = 10e-4  # 學習率
adm = Adam(lr=lr, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
optimizer = adm
lower_lr_limit = 10e-6  # 學習率下限
epoch_down = 100  # 幾次以後開始下降
lr_down = 0.9  # lr 一次下降多少

name = 'create_custom_model1'

model = create_custom_model()

folder = f'./save/{name}'
# Create directories for saving weights
os.makedirs(folder, exist_ok=True)


# model.build(input_shape=(32, 32, 3))  # 請根據你的輸入形狀調整這裡
# Model summary
with open(f'./save/{name}/{name}_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()


# Model weights path
best_acc_loss_path = f'{folder}/best_{name}_acc_loss_weights01.h5'


# Load previous weights if available
if os.path.exists(best_acc_loss_path):
    model.load_weights(best_acc_loss_path)
    print("Loading Successful.")
else:
    print("No weight file")

# Save best weights based on accuracy
if not os.path.exists(best_acc_loss_path):
    model.save_weights(best_acc_loss_path)

# 設定停止點
earlystop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)


# Train the model
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])


# Define ModelCheckpoint to save best weights based on validation loss and accuracy
checkpoint = ModelCheckpoint(f'{folder}/best_{name}_val_loss_weights01.h5', save_best_only=True,
                             save_weights_only=True, monitor='val_loss', mode='min', verbose=1)
accuracy_checkpoint = ModelCheckpoint(f'{folder}/best_{name}_val_acc_weights01.h5',
                                      save_best_only=True, save_weights_only=True, monitor='val_accuracy', mode='max', verbose=1)

# Define LearningRateScheduler to reduce learning rate by 10% after every 10 epochs
# if epoch % 10 == 0 and epoch:
#     return lr * 0.9
# else:
#     return lr

# lr_callback = LearningRateScheduler(lr_scheduler)
# callbacks = [checkpoint, accuracy_checkpoint, lr_callback]


# plt.show()
# Define LearningRateScheduler to plot learning rate
# def plot_learning_rate(epoch, lr):
#     plt.scatter(epoch, lr)
#     plt.xlabel('Epoch')
#     plt.ylabel('Learning Rate')
#     plt.title('Learning Rate')
#     # plt.show()
#     plt.savefig('./save/save_cnn/fig/learning_rate.png')
#     return lr
# lr_scheduler = LearningRateScheduler(plot_learning_rate)
# print("Learning Rate Scheduler", lr_scheduler)

def adjust_learning_rate(epoch, current_lr):
    """
    自定義的學習率調整函數，可根據需要進行調整。

    Parameters:
    - epoch: 目前的訓練周期
    - current_lr: 當前的學習率

    Returns:
    - new_lr: 調整後的學習率
    """
    # 在這裡定義你的學習率調整策略
    if epoch % epoch_down == 0 and epoch != 0:
        new_lr = current_lr * lr_down  # 每 10 個周期減少 10%
    elif current_lr < lower_lr_limit:
        new_lr = lower_lr_limit
    else:
        new_lr = current_lr

    return new_lr


# 定義 LearningRateScheduler
lr_callback = LearningRateScheduler(
    lambda epoch, lr: adjust_learning_rate(epoch, lr))

# 定義TensorBoard回調函數
tensorboard_callback = TensorBoard(
    log_dir=f'./logs/{name}', histogram_freq=1, write_graph=True, write_images=False)

callbacks = [checkpoint, accuracy_checkpoint,
             tensorboard_callback, lr_callback, earlystop]

augmented_data_gen = datagen.flow(x_train, y_train, batch_size=batch_size)

# # 在 model.fit 時使用 callbacks 參數
history = model.fit(augmented_data_gen, batch_size=batch_size, shuffle=shuffle,
                    epochs=epochs_num, validation_data=(x_test, y_test), callbacks=callbacks)


# LR Scheduler
# plt.plot.history = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.savefig('./save/save_cnn/fig/lr.png')

# Train the model with validation data
# model.fit(x_train, y_train, batch_size=batch_size, shuffle=shuffle, epochs=epochs_num, validation_data=(x_test, y_test), callbacks=callbacks)


# # 繪製 loss 下降曲線
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# # 繪製 accuracy 和 validation accuracy 成長圖
# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('./save/save_cnn/fig/acc_val.png')


# Evaluate model on test data
val_loss, val_acc = model.evaluate(
    x_test, y_test, batch_size=batch_size, verbose=0)

best_val_loss = min(history.history['val_loss'])
best_val_acc = max(history.history['val_accuracy'])
print(f'Best Val_Loss: {best_val_loss}, Best Val_Accuracy: {best_val_acc}')

# Save best weights based on validation loss
model.save_weights(f'{folder}/{name}_final_weights01.h5')
