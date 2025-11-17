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
    """Health check endpoint."""  # C0103: English docstring
    return Response({"status": "Study Sessions API is working"})


@api_view(['GET'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def get_sessions(request):
    """Get all active study sessions."""  # C0103: English docstring
    sessions = StudySession.objects.filter(
        is_active=True,
        scheduled_time__gte=timezone.now()
    ).order_by('scheduled_time')
    serializer = StudySessionSerializer(sessions, many=True)
    return Response(serializer.data)


@api_view(['GET'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def get_my_sessions(request):
    """Get user sessions (created or participating)."""  # C0103: English docstring
    try:
        # User created sessions  # N806: English comments
        created_sessions = StudySession.objects.filter(
            created_by=request.user,
            is_active=True
        )

        # Sessions where user is participant
        participant_sessions = StudySession.objects.filter(
            participants__user=request.user,
            participants__is_active=True,
            is_active=True
        )

        # Combine and remove duplicates
        sessions = (created_sessions | participant_sessions).distinct().order_by('scheduled_time')

        # Add participants info
        sessions_with_participants = []
        for session in sessions:
            session_data = StudySessionSerializer(session).data
            # Add creator flag
            session_data['is_creator'] = session.created_by == request.user
            # Add participant flag
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
    """Create study session."""  # C0103: English docstring
    serializer = CreateStudySessionSerializer(data=request.data)
    if serializer.is_valid():
        try:
            session = serializer.save(created_by=request.user)
            # Automatically add creator as participant
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
    """Join study session."""  # C0103: English docstring
    try:
        session = StudySession.objects.get(id=session_id, is_active=True)
    except StudySession.DoesNotExist:
        return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND) # N806: English message

    # Check if already joined
    if SessionParticipant.objects.filter(session=session, user=request.user).exists():
        return Response({'error': 'Already joined this session'}, status=status.HTTP_400_BAD_REQUEST)

    # Check participants limit
    if session.current_participants_count >= session.max_participants:
        return Response({'error': 'Participants limit reached'}, status=status.HTTP_400_BAD_REQUEST)

    # Check if session hasn't started
    if session.scheduled_time <= timezone.now():
        return Response({'error': 'Cannot join started session' }, status=status.HTTP_400_BAD_REQUEST)

    # Join user to session
    SessionParticipant.objects.create(session=session, user=request.user)

    # Return updated session data
    updated_session = StudySessionSerializer(session).data
    return Response({
        'message': 'Successfully joined session',
        'session': updated_session
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def leave_session(request, session_id):
    """Leave study session."""  # C0103: English docstring
    try:
        participant = SessionParticipant.objects.get(
            session_id=session_id,
            user=request.user,
            is_active=True
        )
    except SessionParticipant.DoesNotExist:
        return Response({'error': 'Not a participant of this session'}, status=status.HTTP_404_NOT_FOUND)

    # Creator cannot leave session (should delete it)
    if participant.session.created_by == request.user:
        return Response({'error': 'Creator cannot leave session'}, status=status.HTTP_400_BAD_REQUEST)

    participant.delete()

    # Return updated session data
    session = StudySession.objects.get(id=session_id)
    updated_session = StudySessionSerializer(session).data
    return Response({
        'message': 'Successfully left session',
        'session': updated_session
    })


@api_view(['DELETE'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def delete_session(request, session_id):
    """Delete study session (creator only)."""  # C0103: English docstring
    try:
        session = StudySession.objects.get(id=session_id, created_by=request.user)
    except StudySession.DoesNotExist:
        return Response({'error': 'Session not found or no permissions'}, status=status.HTTP_404_NOT_FOUND)

    session.is_active = False
    session.save()
    return Response({'message': 'Session deleted'})


@api_view(['GET'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def get_invitations(request):
     """Get user invitations."""  # C0103: English docstring
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
    """Send session invitation."""  # C0103: English docstring
    try:
        session_id = request.data.get('session_id')
        invitee_id = request.data.get('user_id')

        if not session_id or not invitee_id:
            return Response({
                'error': 'session_id and user_id are required'
            }, status=status.HTTP_400_BAD_REQUEST)

        session = StudySession.objects.get(id=session_id, is_active=True)
        invitee = User.objects.get(id=invitee_id)

        # Check if invitation already sent
        if SessionInvitation.objects.filter(session=session, invitee=invitee).exists():
            return Response({
                'error': 'Invitation already sent to this user'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Проверяем, что приглашающий - создатель сессии
        if session.created_by != request.user:
            return Response({
                'error': 'Only session creator can send invitations'
            }, status=status.HTTP_403_FORBIDDEN)

        # Проверяем, что не приглашаем самого себя
        if invitee.id == request.user.id:
            return Response({
                'error': 'Cannot send invitation to yourself'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Create invitation
        invitation = SessionInvitation.objects.create(
            session=session,
            inviter=request.user,
            invitee=invitee,
            status='pending'
        )

        return Response({
            'message': 'Invitation sent successfully',
            'invitation_id': invitation.id,
            'session_title': session.title,
            'invitee_name': invitee.username
        }, status=status.HTTP_201_CREATED)

    except StudySession.DoesNotExist:
        return Response({
            'error': 'Session not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except User.DoesNotExist:
        return Response({
            'error': 'User not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        print(f"Error in send_invitation: {e}")
        return Response({
            'error': 'Internal server error'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])  # E306: Added blank lines between functions
@permission_classes([IsAuthenticated])
def respond_to_invitation(request, invitation_id):
    """Respond to invitation."""  # C0103: English docstring
    try:
        invitation = SessionInvitation.objects.get(
            id=invitation_id,
            invitee=request.user,
            status='pending'
        )

        response = request.data.get('response')
        if response not in ['accepted', 'declined']:
            return Response({'error': 'Invalid response'}, status=400)

        invitation.status = response
        invitation.responded_at = timezone.now()
        invitation.save()

        # If accepted - add to participants
        if response == 'accepted':
            # Check participants limit
            if invitation.session.current_participants_count >= invitation.session.max_participants:
                return Response({'error': 'Participants limit reached'}, status=400)

            # Add to participants
            SessionParticipant.objects.get_or_create(
                session=invitation.session,
                user=request.user
            )

        return Response({
            'message': f'Invitation  {response}',
            'invitation_id': invitation.id
        })

    except SessionInvitation.DoesNotExist:
        return Response({'error': 'Invitation not found'}, status=404)
    except Exception as e:
        print(f"Error in respond_to_invitation: {e}")
        return Response({'error': 'Internal server error'}, status=500)


@api_view(['GET'])  # E306: Added blank lines between function
@permission_classes([IsAuthenticated])
def get_session_participants(request, session_id):
    """Get session participants."""  # C0103: English docstring
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