#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <string.h>

typedef struct list_entry_t {
  void *value;
  struct list_entry_t *prev;
  struct list_entry_t *next;
} list_node_t;

typedef struct {
  int count;
  list_node_t *head;
  list_node_t *tail;
  pthread_mutex_t mutex;
} list;

typedef struct thread_data__{
  int pid;
  int tot;
  list *l;
  int *seq;
}thread_data_t;


list *list_create();
void list_free(list *l);

list_node_t *insert(list *l, void *ptr);
int delete(list *l, int number);
void member(list* l, int number);

/* Naive linked list implementation */

list* list_create() {
  list* l = (list*)malloc(sizeof(list));
  l->count = 0;
  l->head = NULL;
  l->tail = NULL;
  pthread_mutex_init(&(l->mutex), NULL);
  return l;
}

void list_free(list* l)
{
  list_node_t *li, *tmp;

  pthread_mutex_lock(&(l->mutex));

  if (l != NULL) {
    li = l->head;
    while (li != NULL) {
      tmp = li->next;
      free(li);
      li = tmp;
    }
  }

  pthread_mutex_unlock(&(l->mutex));
  pthread_mutex_destroy(&(l->mutex));
  free(l);
}

list_node_t* insert(list* l, void* ptr)
{
  list_node_t* li;

  pthread_mutex_lock(&(l->mutex));

  li = (list_node_t*)malloc(sizeof(list_node_t));
  li->value = ptr;
  li->next = NULL;
  li->prev = l->tail;

  if (l->tail == NULL) {
    l->head = l->tail = li;
  } else {
    l->tail->next = li;
    l->tail = li;
  }
  l->count++;

  pthread_mutex_unlock(&(l->mutex));

  return li;
}

int delete(list* l, int number)
{
  int result = 0;
  list_node_t* li = l->head;

  pthread_mutex_lock(&(l->mutex));

  while (li != NULL) {
    if (*(int*)li->value == number) {
      if (li->prev == NULL) {
        l->head = li->next;
      } else {
        li->prev->next = li->next;
      }

      if (li->next == NULL) {
        l->tail = li->prev;
      } else {
        li->next->prev = li->prev;
      }
      l->count--;
      free(li);
      result = 1;
      break;
    }
    li = li->next;
  }

  pthread_mutex_unlock(&(l->mutex));

  return result;
}

void member(list* l, int number)
{
  list_node_t* li;

  pthread_mutex_lock(&(l->mutex));

  li = l->head;
  while (li != NULL) {
    if (*(int*)li->value == number) {
      break;
    }
    li = li->next;
  }

  pthread_mutex_unlock(&(l->mutex));
}

void print_list(list *l) {
  pthread_mutex_lock(&(l->mutex));
  list_node_t *li = l->head;
  while (li != NULL) {
    printf("%d ", *(int*)li->value);
    li = li->next;
  }
  pthread_mutex_unlock(&(l->mutex));
  printf("--\n");

}

int global_step = 0;
pthread_mutex_t gmutex;

void thread_worker(void *arg) {
  thread_data_t *parg = (thread_data_t*)arg;
  int cur;

  while(1) {
    pthread_mutex_lock(&gmutex);
    if(global_step >= parg->tot) {
      pthread_mutex_unlock(&gmutex);
      break;
    }

    cur = parg->seq[global_step];
    ++ global_step;
    pthread_mutex_unlock(&gmutex);


    if(cur == 0) {
      // insert
      int *p = (int*)malloc(sizeof(int));
      *p = rand() % 50;
      insert(parg->l, p);
    } else if (cur == 1) {
      // delete
      delete(parg->l, rand() % 50);
    } else if (cur == 2) {
      // member
      member(parg->l, rand() % 50);
    } else {
      // print
      print_list(parg->l);
    }
  }

}

int main(int argc, char *argv[]) {
  list *test_list = list_create();
  int n = atoi(argv[2]), i;
  int *seq = (int*)malloc(sizeof(int)*n);
  int n_thread = atoi(argv[1]);

  srand(time(NULL));
  for(i = 0; i < n; i ++) {
    // random of 4 ops
    seq[i] = rand() % 4;
  }
  pthread_mutex_init(&gmutex, NULL);
  pthread_t *thread_pool = (pthread_t*)malloc(sizeof(pthread_t) * n);
  thread_data_t *thread_data = (thread_data_t*)malloc(sizeof(thread_data_t) * n_thread);
  for(i = 0; i < n_thread; i ++) {
    thread_data[i].l = test_list;
    thread_data[i].tot = n;
    thread_data[i].seq = seq;
    pthread_create(thread_pool + i, NULL, thread_worker, thread_data + i);
  }

  for(i = 0; i < n_thread; i ++)
    pthread_join(thread_pool[i], NULL);



  pthread_mutex_destroy(&gmutex);
/*
  for(i = 0; i < n; i ++) {
    int *p = (int*)malloc(sizeof(int));
    *p = i*i;
    insert(test_list, p);
    p = (int*)malloc(sizeof(int));
    *p = i*i*i;
    insert(test_list, p);
    delete(test_list, i);
  }
*/
  print_list(test_list);
  free(seq); free(thread_pool); free(thread_data);
  list_free(test_list);
  return 0;
}