# level3_finalproject-nlp-05

<br>

## 📎Fine. (diary-generation-using-chatbot)

![image](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-05/assets/60664644/960ab096-1155-4600-9708-23a296273413)


> 부스트캠프 AI-Tech 5기 NLP 트랙 Level3 Product serving 프로젝트입니다.
> ***Fine.*** 는 소중한 하루를 기록하고 싶지만 시간이 여유롭지 않은 사람들을 대신하여 사용자의 하루를 일기로 생성해주는 App입니다.
> 당일 먹은 음식, 들은 노래, 기억하고 싶은 장면을 챗봇에게 말하거나, 또는 사진을 첨부하여 기록하면, 챗봇이 알맞는 대답을 해줍니다.
> 이후 하루의 마지막에 그 정보를 요약하여 일기를 생성합니다.

<br>

## 🐴Members

|<img src='https://avatars.githubusercontent.com/u/102334596?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/86002769?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/107304584?v=' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/60664644?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/126854237?v=4' height=100 width=100px></img>
| --- | --- | --- | --- | --- |
| [변성훈](https://github.com/DNA-B) | [서보성](https://github.com/Seoboseong) | [이상민](https://github.com/SangMini2) | [이승우](https://github.com/OLAOOT) | [이예원](https://github.com/aeongaewon) |

<br>

## 🤖Model

### PipeLine
![image](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-05/assets/60664644/e673b956-4da2-4cb9-9bd6-00065ff7353b)

<br>

### ChatBot
#### Goal

- ***공감하는*** 리액션과 ***맥락에 맞는*** 질문

#### Data

- `gpt-3.5-turbo` 모델을 이용한 613개의 대화 데이터 생성
    - train(학습) 데이터 개수: 1,696 (83%)
    - test(평가) 데이터 개수: 340 (17%)
    - 대화 데이터 예시
        
        ```
        시스템: 안녕하세요! 오늘 하루는 어떠셨나요? 무슨 재미있는 일이 있었나요?
        사용자: 오늘은 새로운 레스토랑에서 식사를 했어요!
        시스템: 새로운 레스토랑에서 식사를 한다면 항상 기대되는 시간이겠네요! 어떤 음식을 주문해서 먹었나요?
        사용자: 오늘은 일본 음식을 먹을 수 있는 레스토랑에 갔어요. 생선 초밥과 라멘을 주문했는데 정말 맛있었어요!
        시스템: 일본 음식은 그 특별한 맛과 다양한 종류로 유명하죠! 생선 초밥과 라멘이 그 레스토랑에서 어떻게 서비스되었나요? 그 음식을 먹으며 어떤 이야기를 나눴나요?
        사용자: 생선 초밥은 신선하고 맛있는 재료로 정성스럽게 만들어져서 맛있었어요. 라멘은 깊은 맛이 있어서 더욱 좋았어요. 친구와 함께 음식에 대해 이야기하며 서로의 추천 음식을 공유했어요.
        시스템: 신선한 재료로 만들어진 생선 초밥과 깊은 맛이 있는 라멘, 정말 매력적인 메뉴들이었겠네요! 친구와 함께 음식에 대해 이야기하고 추천을 공유하는 건 언제나 즐거운 시간이겠어요. 더 자세히 어떤 음식에 대해 이야기를 나눴나요?
        ```
        

#### Training

- `KULLM` 을 이용한 **Instruction Tuning**과 Parameter-Efficient Fine-Tuning (**PEFT**)
- **input text** (입력): 최대 **3턴**의 대화 history
- **label** (응답): 대화 history 다음에 올 system 응답
- **Prompt**
    
    ```
    아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.
    
    ### 명령어: 주어진 문장들은 이전 대화 내용들입니다. 이에 알맞은 시스템 응답을 만들어주세요.
    
    ### 입력: system: 안녕하세요! 오늘 하루는 어떠셨나요? 무슨 재미있는 일이 있었나요? user: 오늘은 새로운 카페에서 커피를 마셨어요!
    
    ### 응답: system: 좋아요! 새로운 카페에서 커피를 마시면서 어떤 기분이 들었나요? 어떤 종류의 커피를 시도해봤어요?
    ```
    
    - **label** 에 대해서만 loss 계산
    - 58억개 중 약 2억개 parameters를 fine tuning → **사전 학습된 지식 활용**

#### Metric

- Referenece-Based
    - ROUGE-1, ROUGE-SU, Cosine Similarity
- Reference-Free
    - Perplexity, G-Eval

<br>

### Diary Generation
#### Summarization Model

- **Goal**
    - 사용자와 챗봇의 대화 데이터를 이용한 사용자의 하루 요약
    
- **Data**
    - `gpt-3.5-turbo` 모델을 이용한 대화 데이터 생성
    - 생성된 대화 데이터를 `ChatGPT`를 이용해 요약한 요약 데이터 생성
    - train(학습) 데이터 개수: 279 (81.8%)
    - validation(검증) 데이터 개수: 24 (7%)
    - test(시험) 데이터 개수: 38 (11.1%)

- **Model**
    - `gogamza/kobart-summarization`
    
- **Fine-Tuning**
    - Use special token : [USER], [SYSTEM]
    - Input data formatting
    
- **Metric**
    - Rouge score (ROUGE-1, ROUGE-SU)
    

#### Generation Style Transfer Model

- **Goal**
    - 요약된 내용을 부드러운 어투의 일기로 변경
    
- **Model**
    - `NHNDQ/bart-speech-style-converter`
  
<br>

## 🌐 Product Serving
### Front-End
- **React Native** 를 이용한 Cross-Platform App 개발
![image](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-05/assets/60664644/418327b6-838a-4b97-b58a-cb2026c9c054)

<br>

### Back-End
![image](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-05/assets/60664644/ada4f9ef-a16c-4f1c-933b-b80128a8de1e)

<br>

## 📐System Architecture
![image](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-05/assets/60664644/25a50f40-db50-4b89-9856-87065df2f8e7)

<br>

## 🛠️향후 개선 방향
  ### Model
- 챗봇 데이터 생성 방법이 단순해 데이터 다양성 Χ
    - 생성 flow를 재구성해서 응답 유형을 다양화

- KULLM 모델 사이즈로 인해 GPU 서버 사용
    - 유지 비용 비쌈 → **경량화** 필요
    
- Summarization 시, 동일한 내용을 **중복**으로 생성하는 경우가 있음
    - 같은 내용을 두 번 이상 반복하는 것은 ROUGE score에 적극적으로 반영되지 않음
    - **평가 지표** 및 **모델 성능** 개선 필요

<br>

  ### Service
- 대화 내용 감성 분석을 바탕으로 그 날에 어울리는 노래, 미디어 등 **컨텐츠 추천** 기능
- QA Task 적용을 통한 사용자의 문장형 질문에 알맞는 일기 내용을 **검색**하는 기능
- 챗봇 및 일기 **말투 커스터마이징** 기능
- 음성 인식, 결제 내역, 지도 등 다양한 외부 API 연결
- 기타 다양한 기능 추가 및 구독 시스템 등 **수익 모델** 기획을 통해 상업성 도모

<br>

## 🧪Demo
- [youtube](https://youtu.be/5fr-eOwK-7k)

<br>

> 💡 __*프로젝트에 관한 자세한 내용은 [랩업리포트](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-05/blob/main/documents/%5B%EC%B5%9C%EC%A2%85%5DNLP_05_%ED%95%98%EB%A3%A8%20%EC%9A%94%EC%95%BD%20%ED%94%BC%EB%93%9C%20%EC%83%9D%EC%84%B1%20%EC%B1%97%EB%B4%87.pdf) 및 [발표자료](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-05/blob/main/documents/%5B%EC%B5%9C%EC%A2%85%5DNLP_05_%ED%95%98%EB%A3%A8%20%EC%9A%94%EC%95%BD%20%ED%94%BC%EB%93%9C%20%EC%83%9D%EC%84%B1%20%EC%B1%97%EB%B4%87.pdf)를 참고해주세요.*__
