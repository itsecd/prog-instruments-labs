from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.db.models import Q
from django.utils import timezone
from django.contrib.auth.models import User
from .models import StudySession, SessionParticipant, SessionInvitation
from .serializers import StudySessionSerializer, CreateStudySessionSerializer, SessionParticipantSerializer


@api_view(['GET'])  # E301: Added blank lines between imports and code
@permission_classes([AllowAny])
def health_check(request):
    return Response({"status": "Study Sessions API is working"})


@api_view(['GET'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def get_sessions(request):
    """Получить список всех активных учебных сессий"""
    sessions = StudySession.objects.filter(
        is_active=True,
        scheduled_time__gte=timezone.now()
    ).order_by('scheduled_time')
    serializer = StudySessionSerializer(sessions, many=True)
    return Response(serializer.data)


@api_view(['GET'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def get_my_sessions(request):
    """Получить сессии пользователя (созданные или участник)"""
    try:
        # Сессии созданные пользователем
        created_sessions = StudySession.objects.filter(
            created_by=request.user,
            is_active=True
        )

        # Сессии где пользователь участник
        participant_sessions = StudySession.objects.filter(
            participants__user=request.user,
            participants__is_active=True,
            is_active=True
        )

        # Объединяем и убираем дубликаты
        sessions = (created_sessions | participant_sessions).distinct().order_by('scheduled_time')

        # Добавляем информацию об участниках
        sessions_with_participants = []
        for session in sessions:
            session_data = StudySessionSerializer(session).data
            # Добавляем флаг является ли пользователь создателем
            session_data['is_creator'] = session.created_by == request.user
            # Добавляем флаг является ли пользователь участником
            session_data['is_participant'] = session.participants.filter(
                user=request.user,
                is_active=True
            ).exists()
            sessions_with_participants.append(session_data)

        return Response(sessions_with_participants)

    except Exception as e:
        print(f"Error in get_my_sessions: {e}")
        return Response({'error': 'Internal server error'}, status=500)


@api_view(['POST'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def create_session(request):
    """Создать учебную сессию"""
    serializer = CreateStudySessionSerializer(data=request.data)
    if serializer.is_valid():
        try:
            session = serializer.save(created_by=request.user)
            # Автоматически добавляем создателя как участника
            SessionParticipant.objects.create(session=session, user=request.user)

            full_session_data = StudySessionSerializer(session).data
            return Response(full_session_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            print(f"❌ Error creating session: {e}")
            return Response(
                {'error': f'Error creating session: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def join_session(request, session_id):
    """Присоединиться к учебной сессии"""
    try:
        session = StudySession.objects.get(id=session_id, is_active=True)
    except StudySession.DoesNotExist:
        return Response({'error': 'Сессия не найдена'}, status=status.HTTP_404_NOT_FOUND)

    # Проверяем, не присоединен ли уже
    if SessionParticipant.objects.filter(session=session, user=request.user).exists():
        return Response({'error': 'Вы уже присоединены к этой сессии'}, status=status.HTTP_400_BAD_REQUEST)

    # Проверяем лимит участников
    if session.current_participants_count >= session.max_participants:
        return Response({'error': 'Достигнут лимит участников'}, status=status.HTTP_400_BAD_REQUEST)

    # Проверяем, что сессия еще не началась
    if session.scheduled_time <= timezone.now():
        return Response({'error': 'Нельзя присоединиться к начавшейся сессии'}, status=status.HTTP_400_BAD_REQUEST)

    # Присоединяем пользователя
    SessionParticipant.objects.create(session=session, user=request.user)

    # Возвращаем обновленные данные сессии
    updated_session = StudySessionSerializer(session).data
    return Response({
        'message': 'Вы присоединились к сессии',
        'session': updated_session
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def leave_session(request, session_id):
    """Покинуть учебную сессию"""
    try:
        participant = SessionParticipant.objects.get(
            session_id=session_id,
            user=request.user,
            is_active=True
        )
    except SessionParticipant.DoesNotExist:
        return Response({'error': 'Вы не участник этой сессии'}, status=status.HTTP_404_NOT_FOUND)

    # Создатель не может покинуть сессию (должен удалить её)
    if participant.session.created_by == request.user:
        return Response({'error': 'Создатель не может покинуть сессию'}, status=status.HTTP_400_BAD_REQUEST)

    participant.delete()

    # Возвращаем обновленные данные сессии
    session = StudySession.objects.get(id=session_id)
    updated_session = StudySessionSerializer(session).data
    return Response({
        'message': 'Вы покинули сессию',
        'session': updated_session
    })


@api_view(['DELETE'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def delete_session(request, session_id):
    """Удалить учебную сессию (только создатель)"""
    try:
        session = StudySession.objects.get(id=session_id, created_by=request.user)
    except StudySession.DoesNotExist:
        return Response({'error': 'Сессия не найдена или у вас нет прав'}, status=status.HTTP_404_NOT_FOUND)

    session.is_active = False
    session.save()
    return Response({'message': 'Сессия удалена'})


@api_view(['GET'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def get_invitations(request):
    """Получить приглашения пользователя"""
    try:
        invitations = SessionInvitation.objects.filter(
            invitee=request.user
        ).select_related('session', 'inviter', 'inviter__profile')

        invitations_data = []
        for invitation in invitations:
            invitation_data = {
                'id': invitation.id,
                'session': {
                    'id': invitation.session.id,
                    'title': invitation.session.title,
                    'description': invitation.session.description,
                    'subject_name': invitation.session.subject_name,
                    'scheduled_time': invitation.session.scheduled_time,
                    'duration_minutes': invitation.session.duration_minutes,
                    'created_by': invitation.session.created_by.username
                },
                'inviter': {
                    'id': invitation.inviter.id,
                    'username': invitation.inviter.username,
                    'first_name': invitation.inviter.first_name,
                    'last_name': invitation.inviter.last_name
                },
                'status': invitation.status,
                'created_at': invitation.created_at,
                'responded_at': invitation.responded_at
            }
            invitations_data.append(invitation_data)

        return Response(invitations_data)

    except Exception as e:
        print(f"Error in get_invitations: {e}")
        return Response({'error': 'Internal server error'}, status=500)


@api_view(['POST'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def send_invitation(request):
    """Отправить приглашение на сессию"""
    try:
        session_id = request.data.get('session_id')
        invitee_id = request.data.get('user_id')

        if not session_id or not invitee_id:
            return Response({
                'error': 'session_id и user_id обязательны'
            }, status=status.HTTP_400_BAD_REQUEST)

        session = StudySession.objects.get(id=session_id, is_active=True)
        invitee = User.objects.get(id=invitee_id)

        # Проверяем, не отправили ли уже приглашение
        if SessionInvitation.objects.filter(session=session, invitee=invitee).exists():
            return Response({
                'error': 'Приглашение уже отправлено этому пользователю'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Проверяем, что приглашающий - создатель сессии
        if session.created_by != request.user:
            return Response({
                'error': 'Только создатель сессии может отправлять приглашения'
            }, status=status.HTTP_403_FORBIDDEN)

        # Проверяем, что не приглашаем самого себя
        if invitee.id == request.user.id:
            return Response({
                'error': 'Нельзя отправить приглашение самому себе'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Создаем приглашение
        invitation = SessionInvitation.objects.create(
            session=session,
            inviter=request.user,
            invitee=invitee,
            status='pending'
        )

        return Response({
            'message': 'Приглашение успешно отправлено',
            'invitation_id': invitation.id,
            'session_title': session.title,
            'invitee_name': invitee.username
        }, status=status.HTTP_201_CREATED)

    except StudySession.DoesNotExist:
        return Response({
            'error': 'Сессия не найдена'
        }, status=status.HTTP_404_NOT_FOUND)
    except User.DoesNotExist:
        return Response({
            'error': 'Пользователь не найден'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        print(f"Error in send_invitation: {e}")
        return Response({
            'error': 'Внутренняя ошибка сервера'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def respond_to_invitation(request, invitation_id):
    """Ответить на приглашение"""
    try:
        invitation = SessionInvitation.objects.get(
            id=invitation_id,
            invitee=request.user,
            status='pending'
        )

        response = request.data.get('response')
        if response not in ['accepted', 'declined']:
            return Response({'error': 'Неверный ответ'}, status=400)

        invitation.status = response
        invitation.responded_at = timezone.now()
        invitation.save()

        # Если приняли приглашение - добавляем в участники
        if response == 'accepted':
            # Проверяем лимит участников
            if invitation.session.current_participants_count >= invitation.session.max_participants:
                return Response({'error': 'Достигнут лимит участников'}, status=400)

            # Добавляем в участники
            SessionParticipant.objects.get_or_create(
                session=invitation.session,
                user=request.user
            )

        return Response({
            'message': f'Приглашение {response}',
            'invitation_id': invitation.id
        })

    except SessionInvitation.DoesNotExist:
        return Response({'error': 'Приглашение не найдено'}, status=404)
    except Exception as e:
        print(f"Error in respond_to_invitation: {e}")
        return Response({'error': 'Internal server error'}, status=500)


@api_view(['GET'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def get_session_participants(request, session_id):
    """Получить участников сессии"""
    try:
        session = StudySession.objects.get(id=session_id)
        participants = session.participants.filter(is_active=True).select_related('user', 'user__profile')

        participants_data = []
        for participant in participants:
            participant_data = {
                'id': participant.id,
                'user': participant.user.id,
                'user_profile': {
                    'username': participant.user.username,
                    'first_name': participant.user.first_name,
                    'last_name': participant.user.last_name,
                    'faculty': participant.user.profile.faculty if hasattr(participant.user, 'profile') else '',
                    'year_of_study': participant.user.profile.year_of_study if hasattr(participant.user, 'profile') else None
                },
                'joined_at': participant.joined_at
            }
            participants_data.append(participant_data)

        return Response(participants_data)

    except StudySession.DoesNotExist:
        return Response({'error': 'Session not found'}, status=404)