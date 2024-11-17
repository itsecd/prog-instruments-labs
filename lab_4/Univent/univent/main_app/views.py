from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView
import logging

from .forms import LoginUserForm, RegisterUserForm, RegisterPosterForm, SubmitApplicationForm, SignForPosterForm
from .models import Poster, User, Application
from .personal_exceptions import InvalidException

logging.basicConfig(level=logging.INFO, filename="main_app_views.log", filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s")


def index(request):
    logging.info("User requested (GET) index page")
    data = {'posters': Poster.objects.all()}
    return render(request, 'index.html', context=data)


class Posters(ListView):
    # data = {'posters': Poster.objects.all()}
    # return render(request, 'posters.html', context=data)
    logging.info("User requested (GET) Posters page")
    model = Poster
    template_name = 'posters.html'
    context_object_name = 'posters'
    all_posters = Poster.objects.all()
    if not len(all_posters):
        logging.warning("'Posters' doesn't have posts, 'Posters' is empty.")
    extra_context = {
        'posters': Poster.objects.all(),
    }


def poster(request, post_slug):
    post = get_object_or_404(Poster, slug=post_slug)
    if request.method == 'POST':
        try:
            logging.info("User requested (POST) poster page")
            form = SignForPosterForm(request.POST)
            if form.is_valid():
                user = User.objects.get(nickname=request.COOKIES['nickname'])
                post.subscribers.add(user)
                logging.info(f"User '{user}' successfully followed to post.")
                return redirect('personal_account')
        except:
            logging.error("User isn't registered yet.")
            return redirect('registration_page')
    else:
        logging.info("User requested (GET) poster page")
        form = SignForPosterForm()
    context = {'post': post, 'form': form}
    return render(request, 'poster.html', context=context)


def personal_account(request):
    logging.info("User requested (GET) personal_account page")
    try:
        user = User.objects.get(nickname=request.COOKIES['nickname'])
        applications = [application for application in Application.objects.filter(sender=user)]
        user_data = {
            'user': user,
            'created_events': user.created_events.all(),
            'applications': applications,
            'poster_subscribers': Poster.objects.filter(subscribers__nickname=request.COOKIES['nickname']),
        }
        logging.info(f"User '{user}' successfully come to personal account.")
    except:
        logging.warning(f"User tried visit personal account without login.")
        return redirect('login_page')
    return render(request, 'personal_account.html', context=user_data)


def poster_redactor(request):
    if request.method == 'POST':
        logging.info("User requested (POST) poster_redactor page")
        form = RegisterPosterForm(request.POST)
        try:
            user = User.objects.get(nickname=request.COOKIES['nickname'])
            if form.is_valid():
                user.created_events.create(title=form.cleaned_data['title'],
                                   place=form.cleaned_data['place'],
                                   price=form.cleaned_data['price'],
                                   creator=request.COOKIES['nickname'],
                                   short_description=form.cleaned_data['short_description'],
                                   full_description=form.cleaned_data['full_description'],
                                   time_event=form.cleaned_data['time_event'])

                user.save()
                logging.info(f"User '{user}' was register a post successfully.")
                return redirect('posters')
            else:
                logging.warning(f"User '{user}' tried create event with invalid params.")
                raise InvalidException
        except:
            logging.error(f"User '{user}' failed to create a post.")
            form.add_error(None, 'Не удалось создать обьявление')

    else:
        logging.info("User requested (GET) poster_redactor page")
        form = RegisterPosterForm()
    data = {'form': form}
    return render(request, 'poster_redactor.html', context=data)


def submit_application(request):
    # todo - страница с отправлением заявки модераторам
    if request.method == 'POST':
        logging.info("User requested (POST) submit_application page")
        form = SubmitApplicationForm(request.POST)
        user = User.objects.get(nickname=request.COOKIES['nickname'])
        if form.is_valid():
                application = Application(
                    sender=user,
                    title=form.cleaned_data['title'],
                    description=form.cleaned_data['description'],
                    call_time=form.cleaned_data['call_time']
                )
                application.save()
                logging.info(f"Application users '{user}' was submitted successfully.")
                return redirect('personal_account')
        else:
            logging.warning(f"User '{user}' tried submit invalid application.")
    else:
        logging.info("User requested (GET) submit_application page")
        form = SubmitApplicationForm()
    data = {'form': form}
    return render(request, 'submit_application.html', context=data)


def registration_page(request):
    if request.method == 'POST':
        logging.info("User requested (POST) registration_page page")
        form = RegisterUserForm(request.POST)
        if form.is_valid():
            try:
                if User.objects.filter(nickname=form.cleaned_data['nickname']):
                    raise Exception
                user = User(nickname=form.cleaned_data['nickname'],
                            password=form.cleaned_data['password'],
                            name=form.cleaned_data['name'],
                            surname=form.cleaned_data['surname'],
                            age=form.cleaned_data['age'],
                            hobby=form.cleaned_data['hobby'])
                user.save()
                logging.info(f"User '{user}' was register successfully.")
                rsp = redirect('personal_account')
                rsp.set_cookie('nickname', form.cleaned_data['nickname'])
                return rsp
            except:
                logging.warning("User tried to register using existing nickname.")
                form.add_error(None, 'Пользователь с таким ником уже существует')
        else:
            logging.warning("User tried to register with invalid parameters.")
    else:
        logging.info("User requested (GET) registration_page page")
        form = RegisterUserForm()
    data = {'form': form}
    return render(request, 'registration_page.html', context=data)


def login_page(request):
    if request.method == 'POST':
        logging.info("User requested (POST) login_page page")
        form = LoginUserForm(request.POST)
        if form.is_valid():
            if not User.objects.filter(nickname=form.cleaned_data['nickname']):
                logging.info(f"User \'{form.cleaned_data['nickname']}\' is\'t registered.")
                return redirect('registration_page')
            elif User.objects.filter(nickname=form.cleaned_data['nickname'])[0].password \
                    != form.cleaned_data['password']:
                logging.warning("User tried to login with invalid password.")
                form.add_error(None, 'Неправильно указан пароль')
            else:
                try:
                    user = User.objects.get(nickname=request.COOKIES['nickname'])
                    logging.info(f'User {user} successfully login.')
                    return redirect('personal_account')
                except:
                    logging.error(f'User {user} couldn\'t login.')
                    rsp = redirect('index')
                    rsp.set_cookie('nickname', form.cleaned_data['nickname'])
                    return rsp
        else:
            logging.warning("User tried to login with invalid parameters.")

    else:
        logging.info("User requested (GET) login_page page")
        form = LoginUserForm()
    return render(request, 'login_page.html', context={'form': form})
