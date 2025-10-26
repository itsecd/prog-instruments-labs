import java.io.FileWriter;
import java.io.IOException;
import java.security.SecureRandom;


public class RandomSequenceGenerator {
    public static void main(String[] args) {
        // Создание генератора
        SecureRandom random = new SecureRandom();

        // Генерация 128 бит
        byte[] bytes = new byte[16];
        random.nextBytes(bytes);

        // Преобразование массива байтов в двоичную строку
        StringBuilder binaryString = new StringBuilder();
        for (byte b : bytes) {

            String binaryByte = String.format("%8s", Integer.toBinaryString(b & 0xFF)).replace(' ', '0');
            binaryString.append(binaryByte);
        }

        // Запись сгенерированной битовой последовательности в файл
        try (FileWriter writer = new FileWriter("bin_sequence_java.txt")) {
            writer.write(binaryString.toString());
            System.out.println("The sequence is saved in bin_sequence_java.txt");
            System.out.println(binaryString.toString());
        } catch (IOException e) {
            System.err.println("Error when writing to a file: " + e.getMessage());
        }
    }
}