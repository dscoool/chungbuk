# image classification

(2023.01.12 ~ 2023.01.18) 충북대학교 인공지능 이미지 분류 특강

안녕하세요 수강생 여러분? 강사 정재민입니다. 😊

이 강의게시판은 충북대학교 이미지 이미지 분류 특강 (2023.01.12 ~ 2023.01.18) </br>

강의자료 및 공지사항 게시판입니다.

아래 강의게시판의 수업자료를 참조하여 주세요!!

# 수업자료

## [1일차]

### 0. 오리엔테이션 - 강좌 소개

* [PPT] 인공지능 이미지 분류 강좌소개

### 1. 파이썬 환경설정 

이 수업은 Google Colab으로 진행합니다.

* 구글 Colab: http://colab.research.google.com (웹상에서 파이썬 실행)

주피터 노트북(.ipynb) 파일을 연 후, 'Open in Colab'버튼을 누르세요!!

------------------------------------------------------------------------

* 혹시 본인의 PC에서 PyCharm이나 Visual Studio Code를 사용하고자 하시는 분은,

해당 컴파일러를 본인 환경에서 사용하시면 되겠습니다.

컴파일러 옵션 1 = 파이썬 + PyCharm
* 아나콘다: https://www.anaconda.com/products/distribution
* 주피터(Jupyter) 에러시: https://link2me.tistory.com/2094
$ pip install jupyter notebook

컴파일러 옵션 2 = 파이썬 + Visual Studio Code + Jupyter
* 파이썬 다운로드: https://www.python.org/downloads/
* Visual Studio Code: https://code.visualstudio.com/download
* Python + Visual Studio Code + Jupyter 설치 동영상: https://tv.naver.com/v/31561658

### 2. 텐서플로우 예제 1

* 예제: https://github.com/dscoool/chungbuk/blob/main/tensorflow_beginner.ipynb
* 설명서: https://guide.ncloud-docs.com/beta/docs/tensorflow-for-beginner

### 3. 텐서플로우(Tensorflow/Keras) 이미지 분류 예제

* 예제: https://github.com/dscoool/chungbuk/blob/e5a684d46645f0610f34518a1d4bb5d9d4659c93/tensorflow_keras_image_classification.ipynb
* 설명서: https://guide.ncloud-docs.com/beta/docs/tensorflow-keras-image-classification


## [2일차] 

[PPT] https://github.com/dscoool/chungbuk/blob/e7950f6d708b7fda1679ceec5928446326f868f5/%5Bday2%5D%20%EC%B6%A9%EB%B6%81%EB%8C%80%ED%95%99%EA%B5%90%20%ED%8A%B9%EA%B0%95.pdf

안녕하세요 수강생 여러분? 어제는 오리엔테이션을 진행했구요,

오늘은 이제 본격적으로 실습에 들어가보도록 하겠습니다!! 파이썬 랩(lab)이라고 생각하시면 좋을 듯해요!!

우선 머신러닝이 뭔지 감을 익히기 위해, tensorflow에 대해 설명을 들은 뒤 (^^),

코드를 강사와 함께 한 줄씩 입력하도록 하겠습니다!!

실습1과 실습2를 진행할 거구요,

모델을 만든 후 model.predict([10.0]) 명령어를 사용해서

머신러닝을 실제로 사용해 보도록 할게요!!

그럼 시작해 볼까요? 😊


### 실습1

1. 텐서플로우에 대한 설명 (7:17) 을 들은 후에 (한글 자막을 켜세요!!)
https://youtu.be/KNAWp2S3w94

2. 코드를 같이 입력해 봅시다!!
https://developers.google.com/codelabs/tensorflow-1-helloworld#0

3. 같이 코드를 실행해 봅시다!!
https://github.com/dscoool/chungbuk/blob/main/Lab1-Hello-ML-World.ipynb

4. 마지막 줄을 같이 살펴봅시다!!

model.predict([10.0]) 을 실행했을 때, 얼마의 결과가 나오셨나요?

y = 2x – 1 에서 x=10일 때, y를 추론하는 과정입니다. 😊

5. 이렇게 하면 인간의 생각과 컴퓨터의 생각이 비슷하다고 할 수 있을까요?
이제 가위바위보를 떠올리며, 의류 사진을 같은 로직으로 처리해 봅시다!! 😊




### 실습2

이번에는 실습1에서 적용했던 기술을 디지털 이미지 파일에 적용해 봅시다!! 😊

1. 텐서플로우에 대한 설명 (7:23) 을 들은 후에 (한글 자막을 켜세요!!)
https://www.youtube.com/watch?v=bemDFpNooA8

2. 코드를 같이 입력해 봅시다!!
https://github.com/dscoool/chungbuk/blob/main/Lab2-Computer-Vision.ipynb

3. 같이 코드를 실행해 봅시다!!
https://github.com/dscoool/chungbuk/blob/main/Lab2-Computer-Vision.ipynb

4. y=2x-1 을 예측하는 머신러닝 기술을, 이번에는 의류 분류에 사용했습니다.
이를 ‘이미지 분류 기술(classification)이라고 합니다!!

5. 이로서 사람이 의류를 보고 한 눈에 분류 가능하듯이, 컴퓨터로도 이미지를 인식하고 분류할 수 있게 되었습니다!!

## [3일차] 

[PPT]

### 실습3 - 컨볼루션 뉴럴 네트워크 (CNN)

1. 텐서플로우에 대한 설명 (5:32) 을 들은 후에 (한글 자막을 켜세요!!)
https://youtu.be/x_VrgWTKkiM

2. 코드를 같이 입력해 봅시다!!

https://developers.google.com/codelabs/tensorflow-3-convolutions#0

3. 같이 코드를 실행해 봅시다!!

https://github.com/dscoool/chungbuk/blob/main/Lab3-What-Are-Convolutions.ipynb

### 실습4 - 가위바위보 이미지 인식/분류하기!!

1. 텐서플로우에 대한 설명 (5:26) 을 들은 후에 (한글 자막을 켜세요!!)
https://youtu.be/u2TjZzNuly8

2. 가위바위보 게임 설명서를 봅시다 :)
https://www.tensorflow.org/datasets/catalog/rock_paper_scissors

3. 데이터셋을 다운받고 코드를 입력해 실행해 봅시다!!
* 코드(.ipynb) - https://github.com/dscoool/chungbuk/blob/main/Lab4%20-%20RockPaperScissors.ipynb

* 데이터셋

Train set - https://storage.googleapis.com/download.tensorflow.org/data/rps.zip
Test set - https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip


## [과제1] 텐서플로우를 이용하여 이미지 분류 해보기

아래는 학습한 내용을 토대로 하여, 수강생 여러분들 스스로 간단한

인공지능 이미지 분류를 해 보기 위한 과제입니다. 해당 코드를 블로그에 나온

안내에 따라 작성, 실행해 보고, 정리하여 강사에게 이메일로 제출해 주세요!!

수요일 수업에서 코드 리뷰를 진행할 예정입니다.

과제는 A, B 중 하나를 선택하여 진행하시면 됩니다. 

A. 말과 사람 이미지 분류하기
문서 - https://codetorial.net/tensorflow/classifying_the_horse_and_human.html
dataset - https://laurencemoroney.com/datasets.html

B. 고양이와 개 이미지 분류하기
문서 - https://codetorial.net/tensorflow/classifying_the_cats_and_dogs.html
dataset - https://www.kaggle.com/c/dogs-vs-cats

** 해당 과제는 필수가 아니며, 과제를 완료한 후 
수요일에 코드 리뷰(code review)를 진행할 예정입니다!!

-- 제출: 2023/01/18 수업 전까지. jaeminjjeong@kookmin.ac.kr

## [4일차]

### 실습5 - 이미지 처리 기초 - Pillow, Scikit-Image, OpenCV

* 예제: https://github.com/dscoool/chungbuk/blob/main/image_processing.ipynb



### 실습6 - 텍스트 분류

* 예제: https://github.com/dscoool/chungbuk/blob/main/text_classification.ipynb 
* 설명서:https://guide.ncloud-docs.com/beta/docs/tensorflow-keras-text-classification
</br>

## [5일차]

### 코드 리뷰


## [전자신문 인터넷 컨퍼런스 안내] 2023년 2월 24일(금) 10:00~17:20

https://conference.etnews.com/conf_info.html?uid=237

## [참고자료] 이론 - 유튜브 동영상 첨부

* 머신러닝은 어떻게 스스로 학습하는가? (Back Propropagation)

https://www.youtube.com/watch?v=573EZkzfnZ0&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=28

* 신경망의 기본 단위 = 뉴런Neuron
인공신경망의 기본 단위 = MLP (다층 퍼셉트론, Multi Layer Perceptron)

https://www.youtube.com/watch?v=f-EtWNybRoI&list=PLQ28Nx3M4JrhkqBVIXg-i5_CVVoS1UzAv&index=15

