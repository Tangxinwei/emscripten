/*
 * Copyright 2021 The Emscripten Authors.  All rights reserved.
 * Emscripten is available under two separate licenses, the MIT license and the
 * University of Illinois/NCSA Open Source License.  Both these licenses can be
 * found in the LICENSE file.
 */

#include <assert.h>
#include <emscripten/proxying.h>
#include <emscripten/threading.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "em_task_queue.h"
#include "thread_mailbox.h"
#include "proxying_notification_state.h"

// Proxy Queue Lifetime Management
// -------------------------------
//
// Proxied tasks are executed either when the user manually calls
// `emscripten_proxy_execute_queue` on the target thread or when the target
// thread returns to the event loop. The queue does not know which execution
// path will be used ahead of time when the work is proxied, so it must
// conservatively send a message to the target thread's event loop in case the
// user expects the event loop to drive the execution. These notifications
// contain references to the queue that will be dereferenced when the target
// thread returns to its event loop and receives the notification, even if the
// user manages the execution of the queue themselves.
//
// To avoid use-after-free bugs, we cannot free a queue immediately when a user
// calls `em_proxying_queue_destroy`; instead, we have to defer freeing the
// queue until all of its outstanding notifications have been processed. We
// defer freeing the queue using a reference counting scheme. Each time a
// notification containing a reference to the a thread-local task queue is
// generated, we set a flag on that task queue. Each time that task queue is
// processed, we clear the flag. The proxying queue can only be freed once
// `em_proxying_queue_destroy` has been called and the notification flags on
// each of its task queues have been cleared.
//
// But an extra complication is that the target thread may have died by the time
// it gets back to its event loop to process its notifications. This can happen
// when a user proxies some work to a thread, then calls
// `emscripten_proxy_execute_queue` on that thread, then destroys the queue and
// exits the thread. In that situation no work will be dropped, but the thread's
// worker will still receive a notification and have to clear the notification
// flag without a live runtime. Without a live runtime, there is no stack, so
// the worker cannot safely free the queue at this point even if the refcount
// goes to zero. We need a separate thread with a live runtime to perform the
// free.
//
// To ensure that queues are eventually freed, we place destroyed queues in a
// global "zombie list" where they wait for their notification flags to be
// cleared. The zombie list is scanned whenever a new queue is constructed and
// any of the zombie queues without outstanding notifications are freed. In
// principle the zombie list could be scanned at any time, but the queue
// constructor is a nice place to do it because scanning there is sufficient to
// keep the number of zombie queues from growing without bound; creating a new
// zombie ultimately requires creating a new queue.
//
// -------------------------------

struct em_proxying_queue {
  // Protects all accesses to em_task_queues, size, and capacity.
  pthread_mutex_t mutex;
  // `size` task queue pointers stored in an array of size `capacity`.
  em_task_queue** task_queues;
  int size;
  int capacity;
  // Doubly linked list pointers for the zombie list.
  em_proxying_queue* zombie_prev;
  em_proxying_queue* zombie_next;
};

// The system proxying queue.
static em_proxying_queue system_proxying_queue = {.mutex =
                                                    PTHREAD_MUTEX_INITIALIZER,
                                                  .task_queues = NULL,
                                                  .size = 0,
                                                  .capacity = 0,
                                                  .zombie_prev = NULL,
                                                  .zombie_next = NULL};

em_proxying_queue* emscripten_proxy_get_system_queue(void) {
  return &system_proxying_queue;
}

// The head of the zombie list. Its mutex protects access to the list and its
// other fields are not used.
static em_proxying_queue zombie_list_head = {.mutex = PTHREAD_MUTEX_INITIALIZER,
                                             .zombie_prev = &zombie_list_head,
                                             .zombie_next = &zombie_list_head};

static void em_proxying_queue_free(em_proxying_queue* q) {
  pthread_mutex_destroy(&q->mutex);
  for (int i = 0; i < q->size; i++) {
    em_task_queue_destroy(q->task_queues[i]);
  }
  free(q->task_queues);
  free(q);
}

// Does not lock `q` because it should only be called after `q` has been
// destroyed when it would be UB for new work to come in and race to generate a
// new notification.
static int has_notification(em_proxying_queue* q) {
  for (int i = 0; i < q->size; i++) {
    if (q->task_queues[i]->notification != NOTIFICATION_NONE) {
      return 1;
    }
  }
  return 0;
}

static void cull_zombies() {
  pthread_mutex_lock(&zombie_list_head.mutex);
  em_proxying_queue* curr = zombie_list_head.zombie_next;
  while (curr != &zombie_list_head) {
    em_proxying_queue* next = curr->zombie_next;
    if (!has_notification(curr)) {
      // Remove the zombie from the list and free it.
      curr->zombie_prev->zombie_next = curr->zombie_next;
      curr->zombie_next->zombie_prev = curr->zombie_prev;
      em_proxying_queue_free(curr);
    }
    curr = next;
  }
  pthread_mutex_unlock(&zombie_list_head.mutex);
}

em_proxying_queue* em_proxying_queue_create(void) {
  // Free any queue that has been destroyed and is safe to free.
  cull_zombies();

  // Allocate the new queue.
  em_proxying_queue* q = malloc(sizeof(em_proxying_queue));
  if (q == NULL) {
    return NULL;
  }
  *q = (em_proxying_queue){
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .task_queues = NULL,
    .size = 0,
    .capacity = 0,
    .zombie_prev = NULL,
    .zombie_next = NULL,
  };
  return q;
}

void em_proxying_queue_destroy(em_proxying_queue* q) {
  assert(q != NULL);
  assert(q != &system_proxying_queue && "cannot destroy system proxying queue");
  assert(!q->zombie_next && !q->zombie_prev &&
         "double freeing em_proxying_queue!");
  if (!has_notification(q)) {
    // No outstanding references to the queue, so we can go ahead and free it.
    em_proxying_queue_free(q);
    return;
  }
  // Otherwise add the queue to the zombie list so that it will eventually be
  // freed safely.
  pthread_mutex_lock(&zombie_list_head.mutex);
  q->zombie_next = zombie_list_head.zombie_next;
  q->zombie_prev = &zombie_list_head;
  q->zombie_next->zombie_prev = q;
  q->zombie_prev->zombie_next = q;
  pthread_mutex_unlock(&zombie_list_head.mutex);
}

// Not thread safe. Returns NULL if there are no tasks for the thread.
static em_task_queue* get_tasks_for_thread(em_proxying_queue* q,
                                           pthread_t thread) {
  assert(q != NULL);
  for (int i = 0; i < q->size; i++) {
    if (pthread_equal(q->task_queues[i]->thread, thread)) {
      return q->task_queues[i];
    }
  }
  return NULL;
}

// Not thread safe.
static em_task_queue* get_or_add_tasks_for_thread(em_proxying_queue* q,
                                                  pthread_t thread) {
  em_task_queue* tasks = get_tasks_for_thread(q, thread);
  if (tasks != NULL) {
    return tasks;
  }
  // There were no tasks for the thread; initialize a new em_task_queue. If
  // there are not enough queues, allocate more.
  if (q->size == q->capacity) {
    int new_capacity = q->capacity == 0 ? 1 : q->capacity * 2;
    em_task_queue** new_task_queues =
      realloc(q->task_queues, sizeof(em_task_queue*) * new_capacity);
    if (new_task_queues == NULL) {
      return NULL;
    }
    q->task_queues = new_task_queues;
    q->capacity = new_capacity;
  }
  // Initialize the next available task queue.
  tasks = em_task_queue_create(thread);
  if (tasks == NULL) {
    return NULL;
  }
  q->task_queues[q->size++] = tasks;
  return tasks;
}

// Exported for use in worker.js, but otherwise an internal function.
EMSCRIPTEN_KEEPALIVE
void _emscripten_proxy_execute_task_queue(em_task_queue* tasks) {
  // Before we attempt to execute a request from another thread make sure we
  // are in sync with all the loaded code.
  // For example, in PROXY_TO_PTHREAD the atexit functions are called via
  // a proxied call, and without this call to syncronize we would crash if
  // any atexit functions were registered from a side module.
  em_task_queue_execute(tasks);
}

void emscripten_proxy_execute_queue(em_proxying_queue* q) {
  assert(q != NULL);
  assert(pthread_self());

  // Recursion guard to avoid infinite recursion when we arrive here from the
  // pthread_lock call below that executes the system queue. The per-task_queue
  // recursion lock can't catch these recursions because it can only be checked
  // after the lock has been acquired.
  static _Thread_local int executing_system_queue = 0;
  int is_system_queue = q == &system_proxying_queue;
  if (is_system_queue) {
    if (executing_system_queue) {
      return;
    }
    executing_system_queue = 1;
  }

  pthread_mutex_lock(&q->mutex);
  em_task_queue* tasks = get_tasks_for_thread(q, pthread_self());
  pthread_mutex_unlock(&q->mutex);

  if (tasks != NULL && !tasks->processing) {
    // Found the task queue and it is not already being processed; process it.
    em_task_queue_execute(tasks);
  }

  if (is_system_queue) {
    executing_system_queue = 0;
  }
}

static void receive_notification(void* arg) {
  em_task_queue* tasks = arg;
  tasks->notification = NOTIFICATION_RECEIVED;
  em_task_queue_execute(tasks);
  notification_state expected = NOTIFICATION_RECEIVED;
  atomic_compare_exchange_strong(
    &tasks->notification, &expected, NOTIFICATION_NONE);
}

static void cancel_notification(void* arg) {
  em_task_queue* tasks = arg;
  em_task_queue_cancel(tasks);
}

static int do_proxy(em_proxying_queue* q, pthread_t target_thread, task t) {
  assert(q != NULL);
  pthread_mutex_lock(&q->mutex);
  em_task_queue* tasks = get_or_add_tasks_for_thread(q, target_thread);
  pthread_mutex_unlock(&q->mutex);
  if (tasks == NULL) {
    return 0;
  }

  // Ensure the mailbox will remain open or detect that it is already closed.
  if (!emscripten_thread_mailbox_ref(target_thread)) {
    return 0;
  }

  pthread_mutex_lock(&tasks->mutex);
  int enqueued = em_task_queue_enqueue(tasks, t);
  pthread_mutex_unlock(&tasks->mutex);

  if (!enqueued) {
    emscripten_thread_mailbox_unref(target_thread);
    return 0;
  }

  // We're done if there is already a pending notification for this task queue.
  // Otherwise, we will send one.
  notification_state previous =
    atomic_exchange(&tasks->notification, NOTIFICATION_PENDING);
  if (previous == NOTIFICATION_PENDING) {
    emscripten_thread_mailbox_unref(target_thread);
    return 1;
  }

  emscripten_thread_mailbox_send(
    target_thread, (task){receive_notification, cancel_notification, tasks});
  emscripten_thread_mailbox_unref(target_thread);

  return 1;
}

int emscripten_proxy_async(em_proxying_queue* q,
                           pthread_t target_thread,
                           void (*func)(void*),
                           void* arg) {
  return do_proxy(q, target_thread, (task){func, NULL, arg});
}

enum ctx_kind { SYNC, CALLBACK };

enum ctx_state { PENDING, DONE, CANCELED };

struct em_proxying_ctx {
  // The user-provided function and argument.
  void (*func)(em_proxying_ctx*, void*);
  void* arg;

  enum ctx_kind kind;
  union {
    // Context for synchronous proxying.
    struct {
      // Update `state` and signal the condition variable once the proxied task
      // is done or canceled.
      enum ctx_state state;
      pthread_mutex_t mutex;
      pthread_cond_t cond;
    } sync;

    // Context for proxying with callbacks.
    struct {
      em_proxying_queue* queue;
      pthread_t caller_thread;
      void (*callback)(void*);
      void (*cancel)(void*);
    } cb;
  };

  // A doubly linked list of contexts associated with active work on a single
  // thread. If the thread is canceled, it will traverse this list to find
  // contexts that need to be canceled.
  struct em_proxying_ctx* next;
  struct em_proxying_ctx* prev;
};

// The key that `cancel_active_ctxs` is bound to so that it runs when a thread
// is canceled or exits.
static pthread_key_t active_ctxs;
static pthread_once_t active_ctxs_once = PTHREAD_ONCE_INIT;

static void cancel_ctx(void* arg);
static void cancel_active_ctxs(void* arg);

static void init_active_ctxs(void) {
  int ret = pthread_key_create(&active_ctxs, cancel_active_ctxs);
  assert(ret == 0);
  (void)ret;
}

static void add_active_ctx(em_proxying_ctx* ctx) {
  assert(ctx != NULL);
  em_proxying_ctx* head = pthread_getspecific(active_ctxs);
  if (head == NULL) {
    // This is the only active context; initialize the active contexts list.
    ctx->next = ctx->prev = ctx;
    pthread_setspecific(active_ctxs, ctx);
  } else {
    // Insert this context at the tail of the list just before `head`.
    ctx->next = head;
    ctx->prev = head->prev;
    ctx->next->prev = ctx;
    ctx->prev->next = ctx;
  }
}

static void remove_active_ctx(em_proxying_ctx* ctx) {
  assert(ctx != NULL);
  assert(ctx->next != NULL);
  assert(ctx->prev != NULL);
  if (ctx->next == ctx) {
    // This is the only active context; clear the active contexts list.
    ctx->next = ctx->prev = NULL;
    pthread_setspecific(active_ctxs, NULL);
    return;
  }

  // Update the list head if we are removing the current head.
  em_proxying_ctx* head = pthread_getspecific(active_ctxs);
  if (ctx == head) {
    pthread_setspecific(active_ctxs, head->next);
  }

  // Remove the context from the list.
  ctx->prev->next = ctx->next;
  ctx->next->prev = ctx->prev;
  ctx->next = ctx->prev = NULL;
}

static void cancel_active_ctxs(void* arg) {
  pthread_setspecific(active_ctxs, NULL);
  em_proxying_ctx* head = arg;
  em_proxying_ctx* curr = head;
  do {
    em_proxying_ctx* next = curr->next;
    cancel_ctx(curr);
    curr = next;
  } while (curr != head);
}

static void em_proxying_ctx_init_sync(em_proxying_ctx* ctx,
                                      void (*func)(em_proxying_ctx*, void*),
                                      void* arg) {
  pthread_once(&active_ctxs_once, init_active_ctxs);
  *ctx = (em_proxying_ctx){
    .func = func,
    .arg = arg,
    .kind = SYNC,
    .sync =
      {
        .state = PENDING,
        .mutex = PTHREAD_MUTEX_INITIALIZER,
        .cond = PTHREAD_COND_INITIALIZER,
      },
  };
}

static void em_proxying_ctx_init_callback(em_proxying_ctx* ctx,
                                          em_proxying_queue* queue,
                                          pthread_t caller_thread,
                                          void (*func)(em_proxying_ctx*, void*),
                                          void (*callback)(void*),
                                          void (*cancel)(void*),
                                          void* arg) {
  pthread_once(&active_ctxs_once, init_active_ctxs);
  *ctx = (em_proxying_ctx){
    .func = func,
    .arg = arg,
    .kind = CALLBACK,
    .cb =
      {
        .queue = queue,
        .caller_thread = caller_thread,
        .callback = callback,
        .cancel = cancel,
      },
  };
}

static void em_proxying_ctx_deinit(em_proxying_ctx* ctx) {
  if (ctx->kind == SYNC) {
    pthread_mutex_destroy(&ctx->sync.mutex);
    pthread_cond_destroy(&ctx->sync.cond);
  }
  // TODO: We should probably have some kind of refcounting scheme to keep
  // `queue` alive for callback ctxs.
}

static void free_ctx(void* arg) {
  em_proxying_ctx* ctx = arg;
  em_proxying_ctx_deinit(ctx);
  free(ctx);
}

// Free the callback info on the same thread it was originally allocated on.
// This may be more efficient.
static void call_callback_then_free_ctx(void* arg) {
  em_proxying_ctx* ctx = arg;
  ctx->cb.callback(ctx->arg);
  free_ctx(ctx);
}

void emscripten_proxy_finish(em_proxying_ctx* ctx) {
  if (ctx->kind == SYNC) {
    pthread_mutex_lock(&ctx->sync.mutex);
    ctx->sync.state = DONE;
    remove_active_ctx(ctx);
    pthread_mutex_unlock(&ctx->sync.mutex);
    pthread_cond_signal(&ctx->sync.cond);
  } else {
    // Schedule the callback on the caller thread. If the caller thread has
    // already died or dies before the callback is executed, then at least make
    // sure the context is freed.
    remove_active_ctx(ctx);
    if (!do_proxy(ctx->cb.queue,
                  ctx->cb.caller_thread,
                  (task){call_callback_then_free_ctx, free_ctx, ctx})) {
      free_ctx(ctx);
    }
  }
}

static void call_cancel_then_free_ctx(void* arg) {
  em_proxying_ctx* ctx = arg;
  ctx->cb.cancel(ctx->arg);
  free_ctx(ctx);
}

static void cancel_ctx(void* arg) {
  em_proxying_ctx* ctx = arg;
  if (ctx->kind == SYNC) {
    pthread_mutex_lock(&ctx->sync.mutex);
    ctx->sync.state = CANCELED;
    pthread_mutex_unlock(&ctx->sync.mutex);
    pthread_cond_signal(&ctx->sync.cond);
  } else {
    if (ctx->cb.cancel == NULL ||
        !do_proxy(ctx->cb.queue,
                  ctx->cb.caller_thread,
                  (task){call_cancel_then_free_ctx, free_ctx, ctx})) {
      free_ctx(ctx);
    }
  }
}

// Helper for wrapping the call with ctx as a `void (*)(void*)`.
static void call_with_ctx(void* arg) {
  em_proxying_ctx* ctx = arg;
  add_active_ctx(ctx);
  ctx->func(ctx, ctx->arg);
}

int emscripten_proxy_sync_with_ctx(em_proxying_queue* q,
                                   pthread_t target_thread,
                                   void (*func)(em_proxying_ctx*, void*),
                                   void* arg) {
  assert(!pthread_equal(target_thread, pthread_self()) &&
         "Cannot synchronously wait for work proxied to the current thread");
  em_proxying_ctx ctx;
  em_proxying_ctx_init_sync(&ctx, func, arg);
  if (!do_proxy(q, target_thread, (task){call_with_ctx, cancel_ctx, &ctx})) {
    em_proxying_ctx_deinit(&ctx);
    return 0;
  }
  pthread_mutex_lock(&ctx.sync.mutex);
  while (ctx.sync.state == PENDING) {
    pthread_cond_wait(&ctx.sync.cond, &ctx.sync.mutex);
  }
  pthread_mutex_unlock(&ctx.sync.mutex);
  int ret = ctx.sync.state == DONE;
  em_proxying_ctx_deinit(&ctx);
  return ret;
}

// Helper for signaling the end of the task after the user function returns.
static void call_then_finish_sync(em_proxying_ctx* ctx, void* arg) {
  task* t = arg;
  t->func(t->arg);
  emscripten_proxy_finish(ctx);
}

int emscripten_proxy_sync(em_proxying_queue* q,
                          pthread_t target_thread,
                          void (*func)(void*),
                          void* arg) {
  task t = {.func = func, .arg = arg};
  return emscripten_proxy_sync_with_ctx(
    q, target_thread, call_then_finish_sync, &t);
}

static int do_proxy_callback(em_proxying_queue* q,
                             pthread_t target_thread,
                             void (*func)(em_proxying_ctx* ctx, void*),
                             void (*callback)(void*),
                             void (*cancel)(void*),
                             void* arg,
                             em_proxying_ctx* ctx) {
  em_proxying_ctx_init_callback(
    ctx, q, pthread_self(), func, callback, cancel, arg);
  if (!do_proxy(q, target_thread, (task){call_with_ctx, cancel_ctx, ctx})) {
    free_ctx(ctx);
    return 0;
  }
  return 1;
}

int emscripten_proxy_callback_with_ctx(em_proxying_queue* q,
                                       pthread_t target_thread,
                                       void (*func)(em_proxying_ctx* ctx,
                                                    void*),
                                       void (*callback)(void*),
                                       void (*cancel)(void*),
                                       void* arg) {
  em_proxying_ctx* ctx = malloc(sizeof(*ctx));
  if (ctx == NULL) {
    return 0;
  }
  return do_proxy_callback(q, target_thread, func, callback, cancel, arg, ctx);
}

typedef struct callback_ctx {
  void (*func)(void*);
  void (*callback)(void*);
  void (*cancel)(void*);
  void* arg;
} callback_ctx;

static void call_then_finish_callback(em_proxying_ctx* ctx, void* arg) {
  callback_ctx* cb_ctx = arg;
  cb_ctx->func(cb_ctx->arg);
  emscripten_proxy_finish(ctx);
}

static void callback_call(void* arg) {
  callback_ctx* cb_ctx = arg;
  cb_ctx->callback(cb_ctx->arg);
}

static void callback_cancel(void* arg) {
  callback_ctx* cb_ctx = arg;
  if (cb_ctx->cancel != NULL) {
    cb_ctx->cancel(cb_ctx->arg);
  }
}

int emscripten_proxy_callback(em_proxying_queue* q,
                              pthread_t target_thread,
                              void (*func)(void*),
                              void (*callback)(void*),
                              void (*cancel)(void*),
                              void* arg) {
  // Allocate the em_proxying_ctx and the user ctx as a single block.
  struct block {
    em_proxying_ctx ctx;
    callback_ctx cb_ctx;
  };
  struct block* block = malloc(sizeof(*block));
  if (block == NULL) {
    return 0;
  }
  block->cb_ctx = (callback_ctx){func, callback, cancel, arg};
  return do_proxy_callback(q,
                           target_thread,
                           call_then_finish_callback,
                           callback_call,
                           callback_cancel,
                           &block->cb_ctx,
                           &block->ctx);
}
