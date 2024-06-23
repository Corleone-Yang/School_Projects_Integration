#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");
int my_exec(void);
void my_wait(pid_t pid);
int my_fork(void *argc);
static struct task_struct *task;
static struct wait_opts
{
    enum pid_type wo_type;
    int wo_flags;
    struct pid *wo_pid;
    struct siginfo __user *wo_info;
    int __user *wo_stat;
    struct rusage __user *wo_rusage;
    wait_queue_t child_wait;
    int notask_error;
};


extern long _do_fork(unsigned long clone_flags,
                     unsigned long stack_start,
                     unsigned long stack_size,
                     int __user *parent_tidptr,
                     int __user *child_tidptr,
                     unsigned long tls);

extern int do_execve (struct filename *filename,
                      const char __user *const __user *__argv,
                      const char __user *const __user *__envp);

extern long do_wait(struct wait_opts *wo);

extern struct filename *getname(const char __user * filename);

//implement fork function
int my_fork(void *argc){

    //set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	/* fork a process using _do_fork */
	pid_t pid = _do_fork(SIGCHLD,(long unsigned)&my_exec,0,NULL,NULL,0);                  // pay attention

    printk("[program2] : The child process has pid = %ld\n",pid);
    printk("[program2] : This is the parent process, pid = %d\n",(int)current ->pid);
    my_wait(pid);

    return 0;
}


static int __init program2_init(void){

	printk("[program2] : Module_init\n");

    printk("[program2] : Module_init create kthread start\n");
    /* create a kernel thread to run my_fork */

    //create a kthread
    task=kthread_create(&my_fork,NULL,"MyThread");

    //wake up new thread if ok
    if(!IS_ERR(task)){
        printk("[program2] : Module_init kthread starts\n");
        wake_up_process(task);
    }
	
    return 0;
}





// implement exec function

int my_exec(void){
    int result;
    const char path[] = "/opt/test";
    const char *const argv[] = {path,NULL,NULL};
    const char *const envp[] = {"HOME=/","PATH=/sbin:/user/sbin:/bin:/usr/bin",NULL};

    struct filename * my_filename = getname(path);

    result = do_execve(my_filename,argv,envp);

    // if exec success
    if (!result)
        return 0;
    // if exec falied
    do_exit(result);

}

void my_wait(pid_t pid){

    int status;
    struct wait_opts wo;
    struct pid *wo_pid = NULL;
    enum pid_type type;
    type = PIDTYPE_PID;
    wo_pid = find_get_pid(pid);

    wo.wo_type = type;
    wo.wo_pid = wo_pid;
    wo.wo_flags=WEXITED | WUNTRACED;
    wo.wo_info = NULL;
    wo.wo_stat = (int __user*)&status;
    wo.wo_rusage = NULL;

    int a;
    a = do_wait(&wo);

    int sig = *wo.wo_stat;
    switch(sig) {

        case 0:
            printk("[program2] : Child process terminates normally\n");
            break;
        case 1:
            printk("[program2] : Child process gets SIGHUP signal\n");
            printk("[program2] : Child process is hang up\n");
            break;
        case 2:
            printk("[program2] : Child process gets SIGINT signal\n");
            printk("[program2] : Child process is interrupted\n");
            break;
        case 9:
            printk("[program2] : Child process gets SIGKILL signal\n");
            printk("[program2] : Child process is killed\n");
            break;
        case 13:
            printk("[program2] : Child process gets SIGPIPE signal\n");
            printk("[program2] : Child process tries to write a pipe without a process connected to the other end\n");
            break;
        case 14:
            printk("[program2] : Child process gets SIGALRM signal\n");
            printk("[program2] : Child process is alarmed\n");
            break;
        case 15:
            printk("[program2] : Child process gets SIGTERM signal\n");
            printk("[program2] : Child process terminated\n");
            break;
        case 131:
            printk("[program2] : Child process gets SIGQUIT signal\n");
            printk("[program2] : Child process quitted\n");
            break;
        case 132:
            printk("[program2] : Child process gets SIGILL signal\n");
            printk("[program2] : Child process tries to execute an illegal instruction\n");
            break;
        case 133:
            printk("[program2] : Child process gets SIGTRAP signal\n");
            printk("[program2] : Child process is trapped\n");
            break;
        case 134:
            printk("[program2] : Child process gets SIGABRT signal\n");
            printk("[program2] : Child process is aborted\n");
            break;
        case 135:
            printk("[program2] : Child process gets SIGBUS signal\n");
            printk("[program2] : Child process reports a bus error\n");
            break;
        case 136:
            printk("[program2] : Child process gets SIGFPE signal\n");
            printk("[program2] : Child process reports an arithmetic operation error\n");
            break;

        case 139:
            printk("[program2] : Child process gets SIGSEGV signal\n");
            printk("[program2] : Child process makes a segmentation fault\n");
            break;
        case 4991:
            printk("[program2] : Child process gets SIGSTOP signal\n");
            printk("[program2] : Child process stopped\n");
            break;
        default:
            printk("An error occurs. Please check your path\n");

    }


    printk("[program2] : The return signal is %d\n",*wo.wo_stat);
    put_pid(wo_pid);
    return;
}

static void __exit program2_exit(void){
    printk("[program2] : Module_exit\n");
}
module_init(program2_init);
module_exit(program2_exit);

