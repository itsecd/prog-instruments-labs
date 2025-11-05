def luhn_algorithm_check(number_str):
	"""
	Проверяет, соответствует ли число алгоритму Луна.
	Последняя цифра считается контрольной.
	Возвращает True, если номер корректен.
	"""
	total = 0
	# Нумерация идет справа налево, начиная с 1
	for i, char in enumerate(reversed(number_str[:-1]), start=1):
		digit = int(char)
		if i % 2 == 1:
			digit *= 2
			if digit >= 10:
				digit = digit // 10 + digit % 10
		total += digit

	control_digit = int(number_str[-1])
	computed_control = (10 - (total % 10)) % 10
	return computed_control == control_digit


def luhn_algorithm_compute(number_str):
	"""
	Вычисляет контрольную цифру для числа по алгоритму Луна.
	Возвращает число с добавленной контрольной цифрой.
	"""
	total = 0
	for i, char in enumerate(reversed(number_str), start=2):
		digit = int(char)
		if i % 2 == 0:  # Четные позиции (если считать с 1 справа)
			digit *= 2
			if digit >= 10:
				digit = digit // 10 + digit % 10
		total += digit

	control_digit = (10 - (total % 10)) % 10
	return number_str + str(control_digit)
