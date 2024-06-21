import cv2
import albumentations as A
import os

# 이미지 파일 경로 설정
image_paths = ['C:\Robotpro\dataset\\nike\\nike(2).png']

# 증강 파이프라인 정의
transform = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.Flip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Blur(blur_limit=3, p=0.5)
])

# 출력 폴더 생성
output_folder = 'dataset\\augmented_images_nike'
os.makedirs(output_folder, exist_ok=True)

# 각 이미지에 대해 증강된 이미지 생성 및 저장
for image_path in image_paths:
    image = cv2.imread(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

     # 이미지가 제대로 로드되었는지 확인
    if image is None:
        print(f"Error: {image_path} 파일을 읽어올 수 없습니다.")
        continue
    
    for i in range(100):  # 각 이미지당 10개의 변형된 이미지를 생성
        augmented = transform(image=image)['image']
        cv2.imwrite(f'{output_folder}/{image_name}_aug_{i}.jpg', augmented)

print(f'{output_folder} 폴더에 증강된 이미지들이 저장되었습니다.')
