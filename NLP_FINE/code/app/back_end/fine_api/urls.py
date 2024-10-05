from .views import *
from django.urls import path, include
from rest_framework import routers

router = routers.DefaultRouter()
router.register('profile', UserProfileViewSet)
router.register('chat-messages', ChatMessageViewSet, basename='chat-message')
router.register('sum-message', SummarizedMessageViewSet, basename='sum-message')
router.register('image', UploadImage, basename='uploadimage')

urlpatterns = [
    path('', include(router.urls)),
]
