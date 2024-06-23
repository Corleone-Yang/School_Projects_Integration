#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char *argv[]) {

  int status;
  pid_t pid,ppid,cpid;

  /* fork a child process */
  ppid = getpid();
  printf("Process start to fork\n");
  printf("I'm the Parent Process, my pid = %d\n", ppid);

  
  pid = fork();

  if(pid == -1){
    perror("fork fail");
    exit(1);
  }
  /* execute test program */
  else if(pid == 0){
    cpid = getpid();
    printf("I'm the Child Process, my pid = %d\n", cpid);
    printf("Child process start to execute test program:\n");
    char *arg[argc];

    for(int i = 0; i < argc - 1; i++){
      arg[i] = argv[i+1];
    }
    arg[argc - 1] = NULL;
    execve(arg[0], arg, NULL);
  }

  /* wait for child process terminates */

  else {

    waitpid(pid, &status, WUNTRACED);
    printf("Parent process receives SIGCHLD signal\n");


    /* check child process'  termination status */
    if(WIFEXITED(status)){
      printf("Normal termination with EXIT STATUS = %d\n", WEXITSTATUS(status));
    }
    else if (WIFSIGNALED(status)){
      int signal = WTERMSIG(status);

      switch(signal){
        case 1:
          printf("Child process gets SIGHUP signal\n");
          break;
        case 2:
          printf("Child process gets SIGINT signal\n");
          break;
        case 3:
          printf("Child process gets SIGQUIT signal\n");
          break;
        case 4:
          printf("Child process gets SIGILL signal\n");
          break;
        case 5:
          printf("Child process gets SIGTRAP signal\n");
          break;
        case 6:
          printf("Child process gets SIGABRT signal\n");
          break;
        case 7:
          printf("Child process gets SIGBUS signal\n");
          break;
        case 8:
          printf("Child process gets SIGFPE signal\n");
          break;
        case 9:
          printf("Child process gets SIGKILL signal\n");
          break;
        case 11:
          printf("Child process gets SIGSEGV signal\n");
          break;
        case 13:
          printf("Child process gets SIGPIPE signal\n");
          break;
        case 14:
          printf("Child process gets SIGALRM signal\n");
          break;
        case 15:
          printf("Child process gets SIGTERM signal\n");
          break;
        default:
          printf("wrong");
      }
    }

    else if (WIFSTOPPED(status)) {
            if (WSTOPSIG(status) == 17) {                   // pay attention to 17 or 19
                printf("Child process gets SIGSTOP signal\n");
                printf("Child process stopped\n");
            }

            printf("CHILD PROCESS STOPPED");
        }
  }
}
