#ifndef LINKEDLIST_H
#define LINKEDLIST_H

#include <iostream>
#include <random>
#include <stdexcept>

// Шаблонный класс узла двусвязного списка
template <class T>
struct Node {
    T value;       
    Node<T>* next;  
    Node<T>* prev;  
    
    Node() = delete;
    
    Node(T value, Node<T>* next, Node<T>* prev) : value(value), next(next), prev(prev) {}
    
    Node(T value) : value(value), next(nullptr), prev(nullptr) {}
};

// Перегрузка оператора вывода для узла
template <class T>
std::ostream& operator<<(std::ostream& os, const Node<T>& node)
{
    os << node.value << " ";
    return os;
}

// Шаблонный класс двусвязного списка
template <class T>
class LinkedList {
    Node<T>* head_;  
    Node<T>* tail_; 

public:
    LinkedList() : head_(nullptr), tail_(nullptr) {}
    
    LinkedList(const LinkedList& other) {
        Node<T>* tmp = other.head_;  
        if (!tmp)  
        {
            head_ = tail_ = nullptr; 
        }
        else
        {
            // Создаем копию первого узла
            head_ = new Node<T>(tmp->value);
            Node<T>* cur = head_;
            
            // Копируем остальные узлы
            while (tmp != other.tail_)
            {
                tmp = tmp->next;
                cur->next = new Node<T>(tmp->value, nullptr, cur);
                cur = cur->next;
            }
            tail_ = cur;
        }
    }
    
    // Конструктор, создающий список случайных чисел заданной длины
    LinkedList(size_t len, T start, T end, unsigned int seed) {
        if (len == 0)
        {
            head_ = tail_ = nullptr;
        }
        else
        {
            // Инициализация генератора случайных чисел
            std::default_random_engine engine(seed);
            
            // Определяем минимум и максимум без использования std::min/std::max
            double min_val = (static_cast<double>(start) < static_cast<double>(end)) ? 
                             static_cast<double>(start) : static_cast<double>(end);
            double max_val = (static_cast<double>(start) > static_cast<double>(end)) ? 
                             static_cast<double>(start) : static_cast<double>(end);
            
            std::uniform_real_distribution<double> distribution(min_val, max_val);

            // Создаем первый узел
            head_ = new Node<T>(static_cast<T>(distribution(engine)));
            Node<T>* cur = head_;

            // Создаем остальные узлы
            for (size_t i = 0; i < len - 1; i++)
            {
                cur->next = new Node<T>(static_cast<T>(distribution(engine)), nullptr, cur);
                cur = cur->next;
            }
            tail_ = cur;
        }
    }

    // Оператор присваивания с глубоким копированием
    LinkedList& operator=(const LinkedList& rhs)
    {
        // Проверка на самоприсваивание
        if (this != &rhs) {
            // Освобождаем память от текущего списка
            while (head_) {
                Node<T>* temp = head_;
                head_ = head_->next;
                delete temp;
            }
            tail_ = nullptr;
            
            // Копируем данные из rhs
            Node<T>* tmp = rhs.head_;
            if (tmp) {
                head_ = new Node<T>(tmp->value);
                Node<T>* cur = head_;
                while (tmp != rhs.tail_) {
                    tmp = tmp->next;
                    cur->next = new Node<T>(tmp->value, nullptr, cur);
                    cur = cur->next;
                }
                tail_ = cur;
            }
        }
        return *this;
    }

    // Добавление элемента в конец списка
    void push_tail(const T value)
    {
        if (!head_) {
            head_ = tail_ = new Node<T>(value);
            return;
        }
        tail_->next = new Node<T>(value, nullptr, tail_);
        tail_ = tail_->next;
    }

    // Добавление другого списка в конец текущего
    void push_tail(const LinkedList& other)
    {
        if (!other.head_) return;
        
        if (!head_) {  
            // Копируем весь список other
            Node<T>* tmp = other.head_;
            head_ = new Node<T>(tmp->value);
            Node<T>* cur = head_;
            while (tmp != other.tail_) {
                tmp = tmp->next;
                cur->next = new Node<T>(tmp->value, nullptr, cur);
                cur = cur->next;
            }
            tail_ = cur;
        } else {
            // Добавляем узлы из other в конец текущего списка
            Node<T>* tmp = other.head_;
            while (tmp) {
                tail_->next = new Node<T>(tmp->value, nullptr, tail_);
                tail_ = tail_->next;
                tmp = tmp->next;
            }
        }
    }

    // Добавление элемента в начало списка
    void push_head(const T value)
    {
        if (!head_) 
        {
            head_ = tail_ = new Node<T>(value);
            return;
        }
        head_->prev = new Node<T>(value, head_, nullptr);
        head_ = head_->prev;
    }

    // Добавление другого списка в начало текущего
    void push_head(const LinkedList& other)
    {
        if (!other.head_) return; 
        
        // Создаем обратную копию списка other
        LinkedList<T> reversed;
        Node<T>* tmp = other.tail_;
        while (tmp) {
            reversed.push_head(tmp->value); 
            tmp = tmp->prev;
        }
        
        // Соединяем списки
        reversed.tail_->next = head_;
        if (head_) {
            head_->prev = reversed.tail_;
        } else {
            tail_ = reversed.tail_;
        }
        head_ = reversed.head_;
        
        // Предотвращаем удаление узлов при уничтожении reversed
        reversed.head_ = reversed.tail_ = nullptr;
    }

    // Удаление первого элемента списка
    void pop_head()
    {
        if (!head_)  
        {
            return; 
        }
        if (!head_->next) 
        {
            delete head_; 
            head_ = tail_ = nullptr;
            return;
        }
        head_ = head_->next; 
        delete head_->prev;
        head_->prev = nullptr; 
    }

    // Удаление последнего элемента списка
    void pop_tail()
    {
        if (!head_)  
        {
            return; 
        }
        if (!head_->next)  
        {
            delete head_; 
            head_ = tail_ = nullptr;
            return;
        }
        tail_ = tail_->prev;
        delete tail_->next;
        tail_->next = nullptr;
    }

    // Удаление всех узлов с заданным значением
    void delete_node(const T value)
    {
        if (!head_) return;  

        // Удаляем все вхождения в начале списка
        while (head_ && head_->value == value) {
            pop_head();
        }
        
        if (!head_) return; 

        // Проходим по остальным узлам
        Node<T>* cur = head_;
        while (cur->next) {
            if (cur->next->value == value) {
                Node<T>* to_delete = cur->next;
                cur->next = to_delete->next;
                if (to_delete->next) {
                    to_delete->next->prev = cur;
                } else {
                    tail_ = cur; 
                }
                delete to_delete;
            } else {
                cur = cur->next;
            }
        }
    }

    // Константный оператор доступа по индексу (только для чтения)
    T operator[](const size_t index) const
    {
        if (!head_)
        {
            throw std::out_of_range("EMPTY");
        }
        Node<T>* cur = head_; 

        // Проходим по списку до нужного индекса
        for (size_t i = 0; i < index; i++)
        {
            if (!cur->next)  
            {
                throw std::out_of_range("Out of range");
            }
            cur = cur->next;
        }
        return cur->value;
    }
    
    // Не константный оператор доступа по индексу (для чтения и записи)
    T& operator[](const size_t index)
    {
        if (!head_)
        {
            throw std::out_of_range("EMPTY");
        }
        Node<T>* cur = head_; 

        for (size_t i = 0; i < index; i++)
        {
            if (!cur->next)
            {
                throw std::out_of_range("Out of range");
            }
            cur = cur->next;
        }
        return cur->value;
    }

    // Метод для получения размера списка
    size_t size() const
    {
        size_t count = 0;
        Node<T>* cur = head_;
        while (cur) {
            ++count;
            cur = cur->next;
        }
        return count;
    }
    
    // Деструктор - освобождает всю выделенную память
    ~LinkedList()
    {
        while (head_)
        {
            Node<T>* temp = head_;
            head_ = head_->next;
            delete temp;
        }
        tail_ = nullptr;
    }

    friend Stats bubble_sort(LinkedList<int>& arr) {
        Stats s;

        Node<int>* head = arr.head_;
        if (!head) return s;

        bool swapped;
        Node<int>* end = nullptr;

        do {
            swapped = false;
            Node<int>* cur = head;
            while (cur->next != end) {
                s.comparison_count++;
                if (cur->value > cur->next->value) {
                    int temp = cur->value;
                    cur->value = cur->next->value;
                    cur->next->value = temp;
                    s.copy_count += 3;
                    swapped = true;
                }
                cur = cur->next;
            }
            end = cur;
        } while (swapped);

        return s;
    }

    template <class U>
    friend std::ostream& operator<<(std::ostream& os, const LinkedList<U>& list);
};

// Перегрузка оператора вывода для всего списка
template <class T>
std::ostream& operator<<(std::ostream& os, const LinkedList<T>& list)
{
    Node<T>* cur = list.head_;
    while (cur) {
        os << cur->value << " ";
        cur = cur->next;
    }
    os << "\n";
    return os;
}

#endif