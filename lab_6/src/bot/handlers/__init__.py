from .main_menu import router as main_menu_router
from .honeypot import router as honeypot_router


ROUTERS = (
    main_menu_router,
    honeypot_router,
)
