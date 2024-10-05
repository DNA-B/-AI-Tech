from rest_framework import serializers
from .models import *

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        exclude = ['password']

class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ('id', 'message', 'created_at', 'user', 'sender', 'start_chat', 'is_generated')

    def create(self, validated_data):
        if validated_data['message'][:5] == "BOT: ":
            validated_data['message'] = validated_data['message'][5:]
            validated_data['sender'] = 'bot'
        else:
            validated_data['message'] = validated_data['message'][6:]
            validated_data['sender'] = 'user'

        return ChatMessage.objects.create(**validated_data)

class SummarizedMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = SummarizedMessage
        fields = ('id', 'created_at', 'user', 'stylechangemessage', 'start_time')

    def create(self, validated_data):
        return SummarizedMessage.objects.create(**validated_data)

class ImageModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageModel
        fields = ('id', 'created_at', 'user', 'image_link', 'start_time')

    def create(self, validated_data):
        return ImageModel.objects.create(**validated_data)
