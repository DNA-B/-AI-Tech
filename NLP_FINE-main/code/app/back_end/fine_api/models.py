from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.conf import settings
from django.utils import timezone

class UserProfileManager(BaseUserManager):
    def create_user(self, user_id, **kwargs):
        user = self.model(
            user_id=user_id
        )

        user.save(using=self._db)
        
        return user

    def create_superuser(self, user_id, password, **extra_fields):
        validated_data = {'user_id': user_id}
        superuser = UserProfile.objects.create(**validated_data)
        superuser.set_password(password)

        superuser.is_staff = True
        superuser.is_superuser = True
        superuser.is_active = True

        superuser.save(using=self._db)
        
        return superuser

class UserProfile(AbstractBaseUser, PermissionsMixin):
    user_id = models.CharField(max_length=100, unique=True)
    generate_time = models.CharField(max_length=10, default="24")
    is_superuser = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'user_id'

    objects = UserProfileManager()

class ChatMessage(models.Model):
    message = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                                on_delete=models.CASCADE)
    sender = models.CharField(max_length=10, editable=False, default='user')
    start_chat = models.CharField(max_length=10, default='0')
    is_generated = models.BooleanField(default=False, editable=False)

class SummarizedMessage(models.Model):
    created_at = models.DateField(auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                                on_delete=models.CASCADE)
    stylechangemessage = models.CharField(max_length=1000, default='')
    start_time = models.CharField(max_length=100, default='')

class ImageModel(models.Model):
    created_at = models.DateField(auto_now_add=True)
    start_time = models.CharField(max_length=100, default='')
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                                on_delete=models.CASCADE)
    image_link = models.CharField(max_length=5000, default='')

