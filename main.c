#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h> 
#include <unistd.h>
#include <time.h>

int main() {
    int pipe_fd[2];
    if (pipe(pipe_fd) == -1) {
        perror("Pipe failed");
        return 1;
    }

    // 🌍 Timer démarrage
    clock_t start_time = clock();

    pid_t pid = fork();
    if (pid < 0) {
        perror("Fork failed");
        return 1;
    }

    if (pid == 0) {
        // 🧠 Processus fils : entraînement
        printf("[C] Entraînement (partiel) - PID %d | PPID %d\n", getpid(), getppid());
        execlp("python3", "python3", "train_model.py", (char *)NULL);
        perror("execlp failed (train_model.py)");
        exit(1);
    } else {
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            printf("[C] train_model.py terminé avec le code %d\n", WEXITSTATUS(status));
        } else {
            printf("[C] train_model.py ne s'est pas terminé correctement\n");
            return 1;
        }

        // 🔁 Lancer test_model.py
        pid_t pid2 = fork();
        if (pid2 == 0) {
            printf("[C] Test - PID %d | PPID %d\n", getpid(), getppid());
            close(pipe_fd[1]); // Fermer écriture
            char model_path[256];
            read(pipe_fd[0], model_path, sizeof(model_path));
            printf("[C] Modèle à tester : %s\n", model_path);
            execlp("python3", "python3", "test_model.py", model_path, (char *)NULL);
            perror("execlp failed (test_model.py)");
            exit(1);
        } else {
            close(pipe_fd[0]); // Fermer lecture
            const char *model_path = "/home/kali/IA_PROJECT/best_model.keras";
            write(pipe_fd[1], model_path, strlen(model_path) + 1); // +1 pour '\0'

            waitpid(pid2, &status, 0);
            if (WIFEXITED(status)) {
                printf("[C] test_model.py terminé avec le code %d\n", WEXITSTATUS(status));
            } else {
                printf("[C] test_model.py ne s'est pas terminé correctement\n");
                return 1;
            }

            // 🔁 Lancer full_training.py
            pid_t pid3 = fork();
            if (pid3 == 0) {
                printf("[C] Entraînement Complet (train + test) - PID %d | PPID %d\n", getpid(), getppid());
                execlp("python3", "python3", "full_training.py", (char *)NULL);
                perror("execlp failed (full_training.py)");
                exit(1);
            } else {
                waitpid(pid3, &status, 0);
                if (WIFEXITED(status)) {
                    printf("[C] full_training.py terminé avec le code %d\n", WEXITSTATUS(status));
                } else {
                    printf("[C] full_training.py ne s'est pas terminé correctement\n");
                }
            }
        }
    }

    // 🕒 Timer fin
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("[C] Temps total d'exécution : %.2f secondes\n", elapsed_time);

    return 0;
}
