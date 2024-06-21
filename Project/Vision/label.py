import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.utils import to_categorical
import cv2

# 데이터 경로 설정
adidas_path = "C:\\vscode\\robot_programming\\dataset\\augmented_images_adidas"
le_path = "C:\\vscode\\robot_programming\\dataset\\augmented_images_le"
nike_path = "C:\\vscode\\robot_programming\\dataset\\augmented_images_nike"

# 이미지와 라벨 로드
def load_images_and_labels(image_dir, label):
    images = []
    labels = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            img_path = os.path.join(image_dir, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (150, 150))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)
    return images, labels

adidas_images, adidas_labels = load_images_and_labels(adidas_path, 0)
le_images, le_labels = load_images_and_labels(le_path, 1)
nike_images, nike_labels = load_images_and_labels(nike_path, 2)

# 데이터 결합
images = np.array(adidas_images + le_images + nike_images)
labels = np.array(adidas_labels + le_labels + nike_labels)

# 라벨을 원-핫 인코딩
labels = to_categorical(labels, num_classes=3)

# 데이터셋 분할
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# 데이터 증강 설정
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 데이터 증강 적용
train_generator = datagen.flow(X_train, y_train, batch_size=32)
val_generator = datagen.flow(X_val, y_val, batch_size=32)

# CNN 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# 모델 저장
model.save('logo_classification_model.h5')

# 학습 결과 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 예측 예시
img_path = 'C:\\vscode\\robot_programming\\dataset\\augmented_images_le\\le(2)_aug_85.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction[0])

class_labels = {0: 'Adidas', 1: 'Le', 2: 'Nike'}
predicted_label = class_labels[predicted_class]

# 예측 결과 출력
print(f"Predicted class: {predicted_label}")

# 이미지 시각화
plt.imshow(image.load_img(img_path, target_size=(150, 150)))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
