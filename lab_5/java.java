import java.security.SecureRandom;

class Main {
    public static void main(String[] args) {
        // Создание генератора
        SecureRandom random = new SecureRandom();

        // Генерация 128 бит
        byte[] bytes = new byte[16]; // 16 байт * 8 бит/байт = 128 бит
        random.nextBytes(bytes);

        // Преобразование массива байтов в двоичную строку
        StringBuilder binaryString = new StringBuilder();
        for (byte b : bytes) {
            String binaryByte = String.format("%8s", Integer.toBinaryString(b & 0xFF)).replace(' ', '0');
            binaryString.append(binaryByte);
        }

        // Выводим строку в консоль
        System.out.println("Generated Binary Sequence:");
        System.out.println(binaryString.toString());
    }
}