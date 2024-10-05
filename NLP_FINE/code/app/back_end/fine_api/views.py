from rest_framework import viewsets, filters
from .models import *
from .serializers import *
from rest_framework.response import Response
from django.core.exceptions import ObjectDoesNotExist
import torch
from transformers import pipeline, GPTNeoXForCausalLM
from chatbot.utils.prompter import Prompter
import requests
from storages.backends.gcloud import GoogleCloudStorage
from datetime import datetime
import base64
from PIL import Image
from io import BytesIO
from django.core.files.base import ContentFile
from rest_framework.renderers import JSONRenderer

class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer

    def get_queryset(self):
        user_id = self.request.query_params.get('user_id')
        return UserProfile.objects.filter(user_id=user_id)

    def create(self, request, *args, **kwargs):
        data = request.POST.copy()
        
        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data)

class ChatMessageViewSet(viewsets.ModelViewSet):
    serializer_class = ChatMessageSerializer
    
    MODEL = "nlpai-lab/kullm-polyglot-5.8b-v2"
    SAVED_PATH = '/home/sboseong1024/server/chatbot/kullm'
    model = GPTNeoXForCausalLM.from_pretrained(
            SAVED_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
    ).to(device=f"cuda", non_blocking=True)

    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)
    prompter = Prompter("kullm")

    def get_queryset(self):
        user_id = self.request.query_params.get('user_id')

        if user_id:
            try:
                user = UserProfile.objects.get(user_id=user_id)
                return ChatMessage.objects.filter(user=user)
            except ObjectDoesNotExist:
                return ChatMessage.objects.none()
        else:
            return ChatMessage.objects.none()

    def create(self, request, *args, **kwargs):
        post_data = request.POST   
        user = UserProfile.objects.get(user_id=post_data['user_id'])

        if 'start' in post_data:
            user_data = post_data.copy()
            user_data['message'] = "BOT: " + "안녕하세요! 무슨 일이 있으셨나요?"
            user_data['start_chat'] = '1'
            user_data['user'] = user.id
            serializer = self.get_serializer(data=user_data)
            serializer.is_valid(raise_exception=True)
            serializer.save()

            return Response(serializer.data)

        if 'end' in post_data:
            last_chats = ChatMessage.objects.filter(user=user.id).order_by('-created_at')
            last_chat = last_chats[0]

            if last_chats[1].start_chat == "1":
                last_chat.is_generated = True
                last_chats[1].is_generated = True
                last_chats[1].save()

            last_chat.start_chat = "-1"
            last_chat.save()

            return Response("Success")

        user_data = post_data.copy()
        user_data['message'] = "USER: " + user_data['message']
        user_data['user'] = user.id
        serializer = self.get_serializer(data=user_data)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        chat_messages = ChatMessage.objects.filter(user=user.id).order_by('-created_at')
        input_values = []
        cnt = 0

        for chat_message in chat_messages:
            if chat_message.sender == "bot":
                input_values.append("system: " + chat_message.message)

                if chat_message.start_chat == "1":
                    break
            else:
                input_values.append("user: " + chat_message.message)
            
            if cnt == 5:
                break

            cnt += 1

        input_values.reverse()
        input_text = ' '.join(input_values)
        instruction = "주어진 문장들은 이전 대화 내용들입니다. 이에 알맞은 시스템 응답을 만들어주세요."

        prompt = self.prompter.generate_prompt(instruction, input_text)
        output = self.pipe(prompt, max_length=1024, temperature=0.2, num_beams=5, eos_token_id=2)
        s = output[0]["generated_text"]
        bot_message = self.prompter.get_response(s)
        torch.cuda.empty_cache()

        bot_data = post_data.copy()
        bot_data['message'] = "BOT: " + bot_message[8:]
        bot_data['user'] = user.id
        serializer = self.get_serializer(data=bot_data)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data)

class SummarizedMessageViewSet(viewsets.ModelViewSet):
    serializer_class = SummarizedMessageSerializer

    def create(self, request, *args, **kwargs):
        post_data = request.POST

        if 'time' in post_data:
            current_time = post_data['time']

            try:
                users = UserProfile.objects.filter(generate_time=current_time)
            except ObjectDoesNotExist:
                return Response("No object")

            for user in users:
                chat_messages = ChatMessage.objects.filter(user=user.id).order_by('-created_at')
                input_values = []
                tmp = []
                start_times = []

                for chat_message in chat_messages:
                    if chat_message.is_generated:
                        break
                    else:
                        chat_message.is_generated = True

                        if chat_message.sender == 'user':
                            tmp.append('[USER]: ' + chat_message.message)
                        elif chat_message.sender == 'bot' and chat_message.start_chat != '-1':
                            tmp.append('[SYSTEM]: ' + chat_message.message)

                        chat_message.save()

                        if chat_message.start_chat == '1':
                            if len(tmp) == 1:
                                tmp = []
                                continue
                      
                            start_times.append(str(chat_message.created_at))
                            tmp.reverse()
                            input_values.append('<s>' + ''.join(tmp) + '/n###[USER]의 하루를 요약해줘:</s>')
                            tmp = []

                input_values.reverse()
 
                input_data = {
                    'input_values' : input_values
                }

                if len(input_values) == 0:
                    continue

                print(input_data)

                res = requests.post('http://34.64.87.97:8000/api/sum-message/', data=input_data)

                stylechangemessage = eval(res.text)

                for idx in range(len(input_values)):
                    data = {
                        "user" : user.id,
                        "stylechangemessage" : stylechangemessage[idx],
                        "start_time" : start_times[-idx]
                    }

                    serializer = self.get_serializer(data=data)
                    serializer.is_valid(raise_exception=True)
                    serializer.save()

            return Response("success")
        else:
            user_id = post_data['user_id']
            date = post_data['date']

            outputs = []

            user = UserProfile.objects.get(user_id=user_id)
            sum_messages = SummarizedMessage.objects.filter(user=user.id, created_at=date)

            for sum_message in sum_messages:
                try:
                    im = ImageModel.objects.get(start_time=sum_message.start_time)
                    image_link = eval(im.image_link)
                except ObjectDoesNotExist:
                    image_link = []

                outputs.append(
                    {
                        'stylechangemessage' : sum_message.stylechangemessage,
                        'image_link' : image_link,
                        'start_time' : sum_message.start_time
                    }
                )

            return Response(outputs)

class UploadImage(viewsets.ModelViewSet):
    serializer_class = ImageModelSerializer
    storage = GoogleCloudStorage()

    def create(self, request, *args, **kwargs):
        post_data = request.POST

        images = eval(post_data['images'])
        images_urls = []

        for image in images:
            base64_image = base64.b64decode(image)
            image = Image.open(BytesIO(base64_image))

            print(type(image))

            buffer = BytesIO()
            image.save(buffer, format='JPEG')
        
            buffer.seek(0)
        
            target_path = f"image/{datetime.timestamp(datetime.now())}.jpg"
            path = self.storage.save(target_path, ContentFile(buffer.read()))
            images_urls.append(self.storage.url(path))

        user = UserProfile.objects.get(user_id=post_data['user_id'])

        image_data = post_data.copy()
        image_data['image_link'] = str(images_urls)
        image_data['user'] = user.id

        chat_messages = ChatMessage.objects.filter(user=user.id).order_by('-created_at')
        start_time = ''

        for chat_message in chat_messages:
            if chat_message.start_chat == '1':
                start_time = str(chat_message.created_at)
                print(chat_message.message)
                print(chat_message.id)
                break

        res = {
            'user_id' : post_data['user_id'],   
            'urls' : images_urls,
            'start_time' : start_time
        }

        image_data['start_time'] = start_time

        serializer = self.get_serializer(data=image_data)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(res)
