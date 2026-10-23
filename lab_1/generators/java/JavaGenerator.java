import java.util.*;
import java.io.*;

public class JavaGenerator {
    private static final String OUTPUT_FILE = "JavaGenerated.txt";
    private static final int VECTOR_SIZE; // объявление без инициализации

    static {
        int tempSize; // временная переменная
        try {
            tempSize = readConstant("SEQUENCE_LENGTH", "D:\\UniversityLabs\\inform-security-base\\lab_2\\generators\\constants.txt");
        } catch (IOException e) {
            System.err.println("Ошибка чтения константы SEQUENCE_LENGTH из файла: " + e.getMessage());
            e.printStackTrace();
            tempSize = 10; // значение по умолчанию
        }
        // Единственное присваивание final полю
        VECTOR_SIZE = tempSize;
    }

    public static int readConstant(String constantName, String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.startsWith(constantName + " = ")) {
                    String[] parts = line.split(" = ");
                    if (parts.length == 2) {
                        return Integer.parseInt(parts[1].trim());
                    }
                }
            }
        }
        throw new IllegalArgumentException("Constants file not found!");
    }

    // Генерирует один случайный бит (0 или 1)
    public static int generateRandomBit() {
        return Math.random() >= 0.5 ? 1 : 0;
    }

    // Генерирует вектор из VECTOR_SIZE случайных битов
    public static List<Integer> generateRandomVector() {
        List<Integer> vector = new ArrayList<>();
        for (int i = 0; i < VECTOR_SIZE; i++) {
            vector.add(generateRandomBit());
        }
        return vector;
    }

    // Форматирует вектор в строку без разделителей
    private static String formatVector(List<Integer> vector) {
        StringBuilder sb = new StringBuilder();
        for (Integer bit : vector) {
            sb.append(bit);
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_FILE))) {
            List<Integer> randomVector = generateRandomVector();
            String formatted = formatVector(randomVector);
            writer.write(formatted);
            writer.newLine(); // Перенос строки
            System.out.println("Успешно сгенерирован и записан вектор длиной " + VECTOR_SIZE + " в файл " + OUTPUT_FILE);
        } catch (IOException e) {
            System.err.println("Ошибка при записи в файл: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
