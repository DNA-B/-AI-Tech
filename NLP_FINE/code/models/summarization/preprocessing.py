'''
- conversation data를 학습시키기 위한 form으로 바꾸는 작업 코드
예시 ) 

<s>[SYSTEM]: 안녕하세요! 오늘 하루는 어떠셨나요? 무슨 흥미로운 일이 있었나요?
[USER]: 오늘은 독서 클럽 모임에 참여했어요!
[SYSTEM]: 독서 클럽 모임은 정말 좋은 방법이죠! 어떤 책을 읽고 어떤 주제로 이야기를 나눴나요?
[USER]: 저희는 "1984"라는 책을 읽고 토론했어요. 인터넷 감시, 빅 브라더, 독재 등의 주제로 이야기를 나눴는데 정말 흥미로웠어요.
[SYSTEM]: "1984"는 정말로 명작 중 하나죠! 그런 주제들로 토론하면 정말로 시선을 다르게 할 수 있을 것 같아요. 
어떤 의견들이 오갔나요?
[USER]: 의견은 여러 가지로 분분했어요. 어떤 사람들은 개인의 프라이버시가 위협받는다고 생각하고, 
다른 사람들은 사회 안전을 위해서는 감시가 필요하다고 생각했어요.
[SYSTEM]: 정말로 흥미로운 주제들이에요. 서로 다른 의견을 공유하면서 어떤 새로운 시각을 얻었나요?
[USER]: 토론을 통해 다른 사람들의 생각을 들어보니 정말 다양한 관점들이 있음을 깨달았어요. 
서로 다른 배경과 경험을 가진 사람들끼리 얘기하니까 새로운 시각을 얻을 수 있어서 좋았어요.
/n###[USER]의 하루에 대해 요약해줘:</s>

'''

def preprocessing(df):
    all_input_texts = []
    labels = []
    for i in range(len(df)):
        if df.loc[i, "summarizations"] != "error":
            check = eval(df.loc[i, "conversations"])
            all_input = "<s>"

            for line in check:
                if line == check[-1] and line.startswith("시스템:"): continue
                else:
                    if line.startswith("시스템:"):
                        input_text = '[SYSTEM]' + line[3:-1]
                    else:
                        input_text = '[USER]' + line[3:-1]
                    all_input += input_text

            all_input += "/n###[USER]의 하루에 대해 요약해줘:</s>"
            all_input_texts.append(all_input)

            labels.append(df.loc[i, 'summarizations'])

    return all_input_texts, labels