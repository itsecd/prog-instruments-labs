

global main
extern scanf
extern printf
extern exit

section .data
    fmt_in    db "%d", 0
    fmt_out   db "%d ", 0
    fmt_last  db "%d", 10, 0   ; "%d\n"

section .bss
    arr resd 100               ; массив int arr[100]

section .text

main:
    push rbp
    mov rbp, rsp
    sub rsp, 64                ; место для локальных переменных

    ; локальные переменные (смещения от rbp):
    ; [rbp-4]   = n
    ; [rbp-8]   = gap
    ; [rbp-12]  = swapped
    ; [rbp-16]  = i
    ; [rbp-20]  = a
    ; [rbp-24]  = b

    ; === читаем n ===
    lea rdi, [rel fmt_in]      ; "%d"
    lea rsi, [rbp-4]           ; &n
    xor eax, eax
    call scanf

    ; if (n > 100) n = 100;
    mov eax, [rbp-4]
    cmp eax, 100
    jle .n_ok
    mov dword [rbp-4], 100
.n_ok:

    ; === ввод массива ===
    mov dword [rbp-16], 0      ; i = 0
.read_loop:
    mov eax, [rbp-16]
    cmp eax, [rbp-4]           ; i < n ?
    jge .after_read

    lea rdi, [rel fmt_in]
    mov eax, [rbp-16]
    mov ecx, eax
    shl rax, 2
    lea rsi, [rel arr]
    add rsi, rax
    xor eax, eax
    call scanf

    inc dword [rbp-16]
    jmp .read_loop
.after_read:

    ; === comb sort init ===
    mov eax, [rbp-4]
    mov [rbp-8], eax           ; gap = n
    mov dword [rbp-12], 1      ; swapped = 1

.comb_outer:
    ; gap = (gap * 10) / 13
    mov eax, [rbp-8]
    imul eax, 10
    cdq
    mov ecx, 13
    idiv ecx
    mov [rbp-8], eax
    cmp eax, 1
    jge .gap_ok
    mov dword [rbp-8], 1
.gap_ok:

    mov dword [rbp-12], 0      ; swapped = 0
    mov dword [rbp-16], 0      ; i = 0

.loop_i:
    mov eax, [rbp-16]
    add eax, [rbp-8]
    cmp eax, [rbp-4]
    jge .after_loop_i          ; if (i+gap >= n) break

    ; a = arr[i]
    mov eax, [rbp-16]
    mov ecx, eax
    shl rax, 2
    lea rdx, [rel arr]
    add rdx, rax
    mov eax, [rdx]
    mov [rbp-20], eax

    ; b = arr[i+gap]
    mov eax, [rbp-16]
    add eax, [rbp-8]
    mov ecx, eax
    shl rax, 2
    lea rdx, [rel arr]
    add rdx, rax
    mov eax, [rdx]
    mov [rbp-24], eax

    ; if (a > b) swap
    mov eax, [rbp-20]
    cmp eax, [rbp-24]
    jle .no_swap

    ; arr[i] = b
    mov eax, [rbp-16]
    mov ecx, eax
    shl rax, 2
    lea rdx, [rel arr]
    add rdx, rax
    mov eax, [rbp-24]
    mov [rdx], eax

    ; arr[i+gap] = a
    mov eax, [rbp-16]
    add eax, [rbp-8]
    mov ecx, eax
    shl rax, 2
    lea rdx, [rel arr]
    add rdx, rax
    mov eax, [rbp-20]
    mov [rdx], eax

    mov dword [rbp-12], 1      ; swapped = 1
.no_swap:
    inc dword [rbp-16]
    jmp .loop_i
.after_loop_i:

    cmp dword [rbp-8], 1
    jg .comb_outer
    cmp dword [rbp-12], 0
    jne .comb_outer

    ; === печать массива ===
    mov dword [rbp-16], 0
.print_loop:
    mov eax, [rbp-16]
    cmp eax, [rbp-4]
    jge .print_done

    ; arr[i]
    mov eax, [rbp-16]
    mov ecx, eax
    shl rax, 2
    lea rdx, [rel arr]
    add rdx, rax
    mov esi, [rdx]

    ; если последний элемент → печатаем с \n
    mov eax, [rbp-16]
    mov ecx, [rbp-4]
    dec ecx
    cmp eax, ecx
    je .last_elem

    lea rdi, [rel fmt_out]
    xor eax, eax
    call printf
    jmp .after_print
.last_elem:
    lea rdi, [rel fmt_last]
    xor eax, eax
    call printf
.after_print:

    inc dword [rbp-16]
    jmp .print_loop
.print_done:

    mov rsp, rbp
    pop rbp
    mov edi, 0
    call exit
