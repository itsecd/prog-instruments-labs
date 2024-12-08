import pygame.font


class Button():
	"""A class that implements the button"""

	def __init__(self, screen, message):
		"""Initializes the attributes of the button"""
		self.screen = screen
		self.screen_rect = screen.get_rect()
		self.width = 200
		self.height = 50
		self.button_color = (0, 255, 0)
		self.text_color = (255, 255, 255)
		self.font = pygame.font.SysFont(None, 48)
		self.rect = pygame.Rect(0, 0, self.width, self.height)
		self.rect.center = self.screen_rect.center 
		self.prep_msg(message)

	def prep_msg(self, message):
		"""Converts the msg to a rectangle and aligns the text in the center"""
		self.msg_image = self.font.render(message, True, self.text_color,
									 		self.button_color)
		self.msg_image_rect = self.msg_image.get_rect()
		self.msg_image_rect.center = self.rect.center

	def draw_button(self):
		"""Displaying an empty button and displaying a message."""
		self.screen.fill(self.button_color, self.rect)
		self.screen.blit(self.msg_image, self.msg_image_rect)
 