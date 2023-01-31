/*
 * Copyright 2023 The Emscripten Authors.  All rights reserved.
 * Emscripten is available under two separate licenses, the MIT license and the
 * University of Illinois/NCSA Open Source License.  Both these licenses can be
 * found in the LICENSE file.
 */

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

#include "em_task_queue.h"
#include "pthread_impl.h"
#include "thread_mailbox.h"
#include "threading_internal.h"

int emscripten_thread_mailbox_ref(pthread_t thread) {
  // Attempt to increment the refcount, being careful not to increment it if we
  // ever observe a 0.
  int prev_count = thread->mailbox_refcount;
  while (1) {
    if (prev_count == 0) {
      // The mailbox is already closed!
      return 0;
    }
    int desired_count = prev_count + 1;
    if (atomic_compare_exchange_weak(
          &thread->mailbox_refcount, &prev_count, desired_count)) {
      return 1;
    }
  }
}

// Decrement and return the refcount.
void emscripten_thread_mailbox_unref(pthread_t thread) {
  int new_count = atomic_fetch_sub(&thread->mailbox_refcount, 1) - 1;
  assert(new_count >= 0);
  if (new_count == 0) {
    // The count is now zero. The thread that owns this queue may be waiting to
    // shut down. Notify the thread that it is safe to proceed now that the
    // mailbox is closed.
    __builtin_wasm_memory_atomic_notify((int*)&thread->mailbox_refcount, -1);
  }
}

// Defined in emscripten_thread_state.S.
int _emscripten_thread_supports_atomics_wait(void);

void _emscripten_thread_mailbox_shutdown(pthread_t thread) {
  assert(thread == pthread_self());

  // Decrement the refcount and wait for it to reach zero.
  assert(thread->mailbox_refcount > 0);
  int count = atomic_fetch_sub(&thread->mailbox_refcount, 1) - 1;

  while (count != 0) {
    // Wait if possible and otherwise spin.
    if (_emscripten_thread_supports_atomics_wait() &&
        __builtin_wasm_memory_atomic_wait32(
          (int*)&thread->mailbox_refcount, count, -1) == 0) {
      break;
    }
    count = thread->mailbox_refcount;
  }

  // The mailbox is now closed. No more messages will be enqueued. Run the
  // shutdown handler for any message already in the queue.
  em_task_queue_cancel(thread->mailbox);
}

void _emscripten_thread_mailbox_init(pthread_t thread) {
  thread->mailbox = em_task_queue_create(thread);
  thread->mailbox_refcount = 1;
}

void emscripten_thread_mailbox_send(pthread_t thread, task t) {
  assert(thread->mailbox_refcount > 0);

  pthread_mutex_lock(&thread->mailbox->mutex);
  if (!em_task_queue_enqueue(thread->mailbox, t)) {
    assert(0 && "No way to correctly recover from allocation failure");
  }
  pthread_mutex_unlock(&thread->mailbox->mutex);

  em_task_queue_notify(thread->mailbox);
}
