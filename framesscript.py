import cv2

for i in range(1,10):
    video = cv2.VideoCapture(f'veo3videos/video{i}.mp4')

    count = 0
    success, image = video.read()

    while success:

        cv2.imwrite(f'veo3videos/video{i}_frame{count}.jpg',image)

        count +=1
        success, image = video.read()

    video.release()


