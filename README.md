# Kongji-translation
표준어 - 제주어 기계번역

# 제주도 사투리 번역 및 오디오북 생성 프로젝트

## 🏝 프로젝트 개요
사멸되어가는 우리 고유 문화인 제주어를 살리고자 제주도 사투리 번역 및 오디오북 생성 프로젝트를 기획하게 되었다. 기계번역과 음성합성 두 가지 딥러닝 모델을 구현하여 기계번역을 통해 표준어에서 제주어로 번역된 사투리 텍스트를 음성합성을 통해 오디오북을 생성한다.

## 🏝 팀원 소개
<img src="https://github.com/MINJU-KIMmm/GitHubTest/blob/main/image/capstoneTeam.png"/>

## 🏝 System Architecture
<img src="https://github.com/MINJU-KIMmm/GitHubTest/blob/main/image/systemarchitecture.png"/>

## 🏝 TTS
### 1. 데이터
기계번역에는 약 17만개의 말뭉치 데이터(.txt)를 사용한다. 
‘제주어구술자료집’ 2017, 2018년 버전을 병렬적으로 가공한 훈련 데이터 160,356쌍, 검증 데이터 5,000쌍, 테스트 데이터 5,000쌍
https://www.kaggle.com/bryanpark/jit-dataset

### 2. Sentencepiece
BPE 알고리즘과 Unigram 언어 모델을 적용한 Google의 sentencepiece로 텍스트 데이터를 서브워드 단위로 분절하였다.

### 3. Transformer
기계번역 모델로는 meta(구 facebook)이 제공하는 fairseq의 Transformer를 사용할 예정이다.

### 4. 현재 단계
<img src="https://user-images.githubusercontent.com/81293595/144700430-231429a7-a8c3-4044-8e21-a1da595cd5a5.png"/>
<img src="https://user-images.githubusercontent.com/81293595/144700433-5c3b54cb-39e7-40df-aeea-250382edcb06.png"/>


### 5. 프로젝트 진행 상황 및 계획
<img src="https://user-images.githubusercontent.com/81293595/144700381-9192ca32-2964-4f51-99e6-4817f60bb6e3.png"/>

학습 ~1월 9일, 테스트 ~1월 16일

### 6. 기술스택
<img src="https://img.shields.io/badge/Google Colab -F9AB00?style=flat-square&logo=GoogleColab&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Python -3776AB?style=flat-square&logo=Python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter -F37626?style=flat-square&logo=Jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow -181717?style=flat-square&logo=TensorFlow&logoColor=white"/>
<img src="https://img.shields.io/badge/GitHub -181717?style=flat-square&logo=GitHub&logoColor=white"/>

### 7. 현재 폴더 구조
<pre>
<code>
korToJeju
        ├─jit
        │      je.dev
        │      je.test
        │      je.train
        │      ko.dev
        │      ko.test
        │      ko.train
        │      subword_tokenizer_jeju.model
        │      subword_tokenizer_jeju.vocab
        │      subword_tokenizer_kor.model
        │      subword_tokenizer_kor.vocab
        │
        └─jupyter
            │  main.ipynb
            │
            └─.ipynb_checkpoints
                    main-checkpoint.ipynb
</code>
</pre>
