import openai
import time
import pandas as pd
from tqdm import tqdm


def return_response(input_text, prediction):
    # 메시지 설정하기
    guideline = f"""시스템과 유저의 대화 내역(Conversation History)을 받게 될 것입니다. 그리고 대화 내역에 대한 시스템의 응답(Response)이 제시됩니다.
    당신의 작업은 평가 단계에 따라 응답을 평가하는 것입니다.
    이 평가 기준을 꼼꼼히 읽고 이해하는 것이 중요합니다. 평가하는 동안 이 문서를 계속 열어두고 필요할 때 참조해 주세요.
    
    평가 기준:
    - 이해 가능성 (0 - 1): Conversation History에 기반하여 Response를 이해 할 수 있나요?
    - 맥락 유지 (1 - 3): Conversation History를 고려했을 때 Response가 맥락을 유지하나요?
    - 흥미롭기 (1 - 3): Response가 지루한가요, 아니면 흥미로운가요?
    - 전반적인 품질 (1 - 5): 위의 답변을 바탕으로 이 발언의 전반적인 품질에 대한 인상은 어떤가요?
    
    평가 단계:
    1. Conversation History와 Response을 주의깊게 읽습니다.
    2. 위의 평가 기준에 따라 Response을 평가합니다.
    
    Conversation History:
    {input_text}
    
    Response:
    {prediction}
    
    Result (scores ONLY)
    - 이해 가능성 (0 - 1):
    - 맥락 유지 (1 - 3):
    - 흥미롭기 (1 - 3):
    - 전반적인 품질 (1 - 5):"""

    messages = [{
        "role": "user", "content": guideline
    }]

    while True:
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=3,
            )
            break
        except Exception as e:
            print(e)
            time.sleep(60)

    all_response = [completion['choices'][i]['message']['content']
                    for i in range(len(completion['choices']))]

    #scores = {'이해 가능성': 0, '맥락 유지': 0, '흥미롭기': 0, '전반적인 품질': 0}

    # for responses in all_response:
    #    responses = responses.split('\n')
    #    for response in responses:
    #        response = response.split(':')
    #        for key in scores.keys():
    #            if key in response[0]:
    #                scores[key] += float(response[1])
    #                break

    # for key in scores.keys():
    #    scores[key] /= 3

    return all_response  # scores


OPENAI_API_KEY = f"{API_KEY}"

# openai API 키 인증
openai.api_key = OPENAI_API_KEY

# 모델 - GPT 3.5 Turbo 선택
model = "gpt-3.5-turbo"
predictions = pd.read_csv('/opt/ml/ignore_folder/kobart_prediction.csv')

#predictions['이해 가능성'], predictions['맥락 유지'], predictions['흥미롭기'], predictions['전반적인 품질'] = None, None, None, None
predictions['평가1'], predictions['평가2'], predictions['평가3'] = None, None, None

for idx, row in tqdm(predictions.iterrows(), total=len(predictions)):
    #if (idx != 38):
    #    continue
    
    input_text = row['input_texts']
    predict = row['prediction']
    labels = row['labels']

    all_response = return_response(input_text=input_text, prediction=predict)
    predictions.loc[idx, '평가1'] = all_response[0]
    predictions.loc[idx, '평가2'] = all_response[1]
    predictions.loc[idx, '평가3'] = all_response[2]

    #predict_scores = return_response(input_text=input_text, prediction=predict)

    # for key in predict_scores.keys():
    #    predictions.loc[idx, key] = predict_scores[key]

    if idx % 5 == 0:
        predictions.to_csv('kobart_prediction_save.csv')

predictions.to_csv('kobart_prediction_predict.csv')
