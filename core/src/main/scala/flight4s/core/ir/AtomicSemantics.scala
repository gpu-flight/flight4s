package flight4s.core.ir

enum MemoryOrder(val cudaName: String):
  case Relaxed extends MemoryOrder("__NV_ATOMIC_RELAXED")
  case Acquire extends MemoryOrder("__NV_ATOMIC_ACQUIRE")
  case Release extends MemoryOrder("__NV_ATOMIC_RELEASE")
  case AcquireRelease extends MemoryOrder("__NV_ATOMIC_ACQ_REL")
  case SequentiallyConsistent extends MemoryOrder("__NV_ATOMIC_SEQ_CST")

  def validForLoad: Boolean = this != Release && this != AcquireRelease
  def validForStore: Boolean = this != Acquire && this != AcquireRelease

  def permitsFailure(failure: MemoryOrder): Boolean = failure match
    case Relaxed => true
    case Acquire => this == Acquire || this == AcquireRelease || this == SequentiallyConsistent
    case SequentiallyConsistent => this == SequentiallyConsistent
    case Release | AcquireRelease => false

enum AtomicScope(val cudaName: String):
  case Block extends AtomicScope("__NV_THREAD_SCOPE_BLOCK")
  case Device extends AtomicScope("__NV_THREAD_SCOPE_DEVICE")

enum AtomicOperation(val cudaName: String, val operandCount: Int, val integralOnly: Boolean):
  case Load extends AtomicOperation("__nv_atomic_load", 0, false)
  case Exchange extends AtomicOperation("__nv_atomic_exchange", 1, false)
  case FetchAdd extends AtomicOperation("__nv_atomic_fetch_add", 1, false)
  case FetchSub extends AtomicOperation("__nv_atomic_fetch_sub", 1, false)
  case FetchMin extends AtomicOperation("__nv_atomic_fetch_min", 1, true)
  case FetchMax extends AtomicOperation("__nv_atomic_fetch_max", 1, true)
  case FetchAnd extends AtomicOperation("__nv_atomic_fetch_and", 1, true)
  case FetchOr extends AtomicOperation("__nv_atomic_fetch_or", 1, true)
  case FetchXor extends AtomicOperation("__nv_atomic_fetch_xor", 1, true)
  case CompareExchange extends AtomicOperation("__nv_atomic_compare_exchange_n", 2, true)
