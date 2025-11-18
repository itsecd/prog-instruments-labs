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
from .logging_config import (
    auth_logger, profile_logger, files_logger, api_logger,
    log_execution_time, PerformanceMixin
)


# Класс-based view для профиля с PerformanceMixin
class UserProfileView(generics.RetrieveUpdateAPIView, PerformanceMixin):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user

    @log_execution_time('profile')
    def retrieve(self, request, *args, **kwargs):
        profile_logger.info(f"Profile view accessed by user: {request.user.username}")
        return super().retrieve(request, *args, **kwargs)

    @log_execution_time('profile')
    def update(self, request, *args, **kwargs):
        profile_logger.info(f"Profile update initiated by user: {request.user.username}")
        return super().update(request, *args, **kwargs)


# Функция-based views с логированием времени выполнения
@api_view(['POST'])
@permission_classes([AllowAny])
@log_execution_time('auth')
def register(request):
    """Регистрация нового пользователя"""
    try:
        auth_logger.info("Registration request received")

        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)

            auth_logger.info(
                f"User registered successfully: {user.username} "
                f"(ID: {user.id}, Email: {user.email})"
            )

            return Response({
                'user': UserSerializer(user).data,
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)

        auth_logger.warning(
            f"Registration validation failed for username: {request.data.get('username', 'unknown')}. "
            f"Errors: {serializer.errors}"
        )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        auth_logger.error(
            f"Registration error: {str(e)}",
            exc_info=True
        )
        return Response(
            {'error': 'Internal server error during registration'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
@log_execution_time('auth')
def login(request):
    """Аутентификация пользователя"""
    try:
        username = request.data.get('username')
        auth_logger.info(f"Login attempt for username: {username}")

        # Basic validation
        if not username or not request.data.get('password'):
            auth_logger.warning("Login attempt with missing credentials")
            return Response(
                {'error': 'Username and password required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = authenticate(username=username, password=request.data.get('password'))

        if user:
            refresh = RefreshToken.for_user(user)
            auth_logger.info(f"Successful login for user: {username} (ID: {user.id})")

            return Response({
                'user': UserSerializer(user).data,
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            })

        auth_logger.warning(f"Failed login attempt for username: {username}")
        return Response(
            {'error': 'Invalid credentials'},
            status=status.HTTP_401_UNAUTHORIZED
        )

    except Exception as e:
        auth_logger.error(
            f"Login error for username {request.data.get('username')}: {str(e)}",
            exc_info=True
        )
        return Response(
            {'error': 'Internal server error during login'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@log_execution_time('profile')
def get_profile(request):
    """Получить профиль текущего пользователя"""
    try:
        profile_logger.debug(
            f"Profile retrieval for user: {request.user.username} (ID: {request.user.id})"
        )

        user_data = UserSerializer(request.user).data
        profile_logger.info(f"Profile retrieved successfully for user: {request.user.username}")

        return Response(user_data)

    except Exception as e:
        profile_logger.error(
            f"Get profile error for user {request.user.username}: {str(e)}",
            exc_info=True
        )
        return Response(
            {'error': 'Internal server error'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['PUT'])
@permission_classes([IsAuthenticated])
@log_execution_time('profile')
def update_profile(request):
    """Обновление профиля пользователя"""
    try:
        user = request.user
        profile_logger.info(
            f"Profile update request from user: {user.username} (ID: {user.id})"
        )

        # Log update fields
        update_fields = list(request.data.keys())
        profile_logger.debug(f"Update fields requested: {update_fields}")

        profile = user.profile

        # Обновляем данные пользователя
        if 'first_name' in request.data:
            user.first_name = request.data['first_name']
            profile_logger.debug(f"Updating first_name to: {request.data['first_name']}")
        if 'last_name' in request.data:
            user.last_name = request.data['last_name']
            profile_logger.debug(f"Updating last_name to: {request.data['last_name']}")

        user.save()

        # Обновляем профиль
        serializer = UserProfileUpdateSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            updated_user = User.objects.get(id=user.id)

            profile_logger.info(f"Profile updated successfully for user: {user.username}")
            return Response(UserSerializer(updated_user).data)

        profile_logger.warning(
            f"Profile update validation failed for user {user.username}: {serializer.errors}"
        )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        profile_logger.error(
            f"Update profile error for user {request.user.username}: {str(e)}",
            exc_info=True
        )
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(['GET'])
@permission_classes([AllowAny])
@log_execution_time('api')
def get_universities(request):
    """Получить список университетов"""
    try:
        api_logger.debug("Universities list request received")

        universities = University.objects.all()
        serializer = UniversitySerializer(universities, many=True)

        api_logger.info(f"Retrieved {universities.count()} universities")
        return Response(serializer.data)

    except Exception as e:
        api_logger.error(
            f"Get universities error: {str(e)}",
            exc_info=True
        )
        return Response(
            {'error': 'Internal server error'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@log_execution_time('files')
def upload_avatar(request):
    """Загрузка аватарки пользователя"""
    try:
        user = request.user
        files_logger.info(f"Avatar upload request from user: {user.username}")

        if 'avatar' not in request.FILES:
            files_logger.warning("Avatar upload attempt without file")
            return Response(
                {'error': 'Файл не найден'},
                status=status.HTTP_400_BAD_REQUEST
            )

        avatar_file = request.FILES['avatar']
        files_logger.debug(
            f"Avatar file details: {avatar_file.name}, "
            f"size: {avatar_file.size} bytes, "
            f"content_type: {avatar_file.content_type}"
        )

        # Validate file size (max 5MB)
        if avatar_file.size > 5 * 1024 * 1024:
            files_logger.warning(f"Avatar file too large: {avatar_file.size} bytes")
            return Response(
                {'error': 'File size exceeds 5MB limit'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Создаем или обновляем аватар
        avatar, created = UserAvatar.objects.get_or_create(user=request.user)

        if created:
            files_logger.debug(f"Created new avatar record for user: {user.username}")
        else:
            files_logger.debug(f"Updating existing avatar for user: {user.username}")

        # Удаляем старый файл если есть
        if avatar.image:
            files_logger.debug("Deleting old avatar file")
            avatar.image.delete(save=False)

        avatar.image.save(avatar_file.name, ContentFile(avatar_file.read()))
        avatar.save()

        files_logger.info(f"Avatar uploaded successfully for user: {user.username}")
        serializer = UserAvatarSerializer(avatar)
        return Response(serializer.data)

    except Exception as e:
        files_logger.error(
            f"Avatar upload error for user {request.user.username}: {str(e)}",
            exc_info=True
        )
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
@log_execution_time('files')
def delete_avatar(request):
    """Удаление аватарки пользователя"""
    try:
        user = request.user
        files_logger.info(f"Avatar deletion request from user: {user.username}")

        avatar = UserAvatar.objects.get(user=request.user)
        avatar.image.delete(save=True)
        avatar.delete()

        files_logger.info(f"Avatar deleted successfully for user: {user.username}")
        return Response({'message': 'Аватар удален'})

    except UserAvatar.DoesNotExist:
        files_logger.warning(f"Avatar deletion attempted but no avatar found for user: {user.username}")
        return Response(
            {'error': 'Аватар не найден'},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        files_logger.error(
            f"Avatar deletion error for user {user.username}: {str(e)}",
            exc_info=True
        )
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
@log_execution_time('api')
def health_check(request):
    """Health check endpoint"""
    api_logger.debug("Health check request received")
    return Response({"status": "Users API is working"})


# Middleware для логирования всех запросов
class RequestLoggingMiddleware:
    """
    Middleware для логирования входящих запросов и исходящих ответов.
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.logger = logging.getLogger('users_api')

    def __call__(self, request):
        import time
        start_time = time.time()

        # Логируем входящий запрос
        self.logger.info(
            f"REQUEST - {request.method} {request.path} - "
            f"User: {getattr(request.user, 'username', 'Anonymous')} - "
            f"IP: {self.get_client_ip(request)}"
        )

        response = self.get_response(request)

        # Логируем исходящий ответ
        execution_time = time.time() - start_time
        self.logger.info(
            f"RESPONSE - {request.method} {request.path} - "
            f"Status: {response.status_code} - "
            f"Time: {execution_time:.3f}s - "
            f"User: {getattr(request.user, 'username', 'Anonymous')}"
        )

        return response

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip