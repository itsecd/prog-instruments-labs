import logging
from rest_framework import status, generics, permissions
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.conf import settings
from .models import UserProfile, University, UserAvatar
from .serializers import (
    UserRegistrationSerializer, UserSerializer,
    UserProfileUpdateSerializer, UniversitySerializer,
    UserAvatarSerializer
)
from .logging_config import api_logger


# Класс-based view для профиля
class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user


# Функция-based views
@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    """Регистрация нового пользователя"""
    try:
        api_logger.info("Registration request received")

        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)

            api_logger.info(f"User registered successfully: {user.username}")

            return Response({
                'user': UserSerializer(user).data,
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)

        api_logger.warning(f"Registration validation failed: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        api_logger.error(f"Registration error: {str(e)}")
        return Response(
            {'error': 'Internal server error during registration'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    """Аутентификация пользователя"""
    try:
        username = request.data.get('username')
        api_logger.info(f"Login attempt for username: {username}")

        user = authenticate(username=username, password=request.data.get('password'))

        if user:
            refresh = RefreshToken.for_user(user)
            api_logger.info(f"Successful login for user: {username}")

            return Response({
                'user': UserSerializer(user).data,
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            })

        api_logger.warning(f"Failed login attempt for username: {username}")
        return Response(
            {'error': 'Invalid credentials'},
            status=status.HTTP_401_UNAUTHORIZED
        )

    except Exception as e:
        api_logger.error(f"Login error: {str(e)}")
        return Response(
            {'error': 'Internal server error during login'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_profile(request):
    """Получить профиль текущего пользователя"""
    try:
        api_logger.info(f"Profile retrieval for user: {request.user.username}")

        user_data = UserSerializer(request.user).data
        return Response(user_data)

    except Exception as e:
        api_logger.error(f"Get profile error: {str(e)}")
        return Response(
            {'error': 'Internal server error'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    """Обновление профиля пользователя"""
    try:
        user = request.user
        api_logger.info(f"Profile update request from user: {user.username}")

        profile = user.profile

        # Обновляем данные пользователя
        if 'first_name' in request.data:
            user.first_name = request.data['first_name']
        if 'last_name' in request.data:
            user.last_name = request.data['last_name']

        user.save()

        # Обновляем профиль
        serializer = UserProfileUpdateSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            updated_user = User.objects.get(id=user.id)

            api_logger.info(f"Profile updated successfully for user: {user.username}")
            return Response(UserSerializer(updated_user).data)

        api_logger.warning(f"Profile update validation failed: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        api_logger.error(f"Update profile error: {str(e)}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def get_universities(request):
    """Получить список университетов"""
    try:
        api_logger.debug("Universities list request received")

        universities = University.objects.all()
        serializer = UniversitySerializer(universities, many=True)
        return Response(serializer.data)

    except Exception as e:
        api_logger.error(f"Get universities error: {str(e)}")
        return Response(
            {'error': 'Internal server error'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_avatar(request):
    """Загрузка аватарки пользователя"""
    try:
        user = request.user
        api_logger.info(f"Avatar upload request from user: {user.username}")

        if 'avatar' not in request.FILES:
            api_logger.warning("Avatar upload attempt without file")
            return Response(
                {'error': 'Файл не найден'},
                status=status.HTTP_400_BAD_REQUEST
            )

        avatar_file = request.FILES['avatar']

        # Создаем или обновляем аватар
        avatar, created = UserAvatar.objects.get_or_create(user=request.user)

        # Удаляем старый файл если есть
        if avatar.image:
            avatar.image.delete(save=False)

        avatar.image.save(avatar_file.name, ContentFile(avatar_file.read()))
        avatar.save()

        api_logger.info(f"Avatar uploaded successfully for user: {user.username}")
        serializer = UserAvatarSerializer(avatar)
        return Response(serializer.data)

    except Exception as e:
        api_logger.error(f"Avatar upload error: {str(e)}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_avatar(request):
    """Удаление аватарки пользователя"""
    try:
        user = request.user
        api_logger.info(f"Avatar deletion request from user: {user.username}")

        avatar = UserAvatar.objects.get(user=request.user)
        avatar.image.delete(save=True)
        avatar.delete()

        api_logger.info(f"Avatar deleted successfully for user: {user.username}")
        return Response({'message': 'Аватар удален'})

    except UserAvatar.DoesNotExist:
        api_logger.warning(f"Avatar deletion attempted but no avatar found for user: {user.username}")
        return Response(
            {'error': 'Аватар не найден'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        api_logger.error(f"Avatar deletion error: {str(e)}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint"""
    api_logger.debug("Health check request received")
    return Response({"status": "Users API is working"})