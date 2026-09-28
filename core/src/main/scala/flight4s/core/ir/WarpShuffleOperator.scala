package flight4s.core.ir

import flight4s.core.types.{CudaType, I32, U32, UInt}

enum WarpShuffleOperator[S](val selectorType: CudaType[S], val cudaName: String, val selectorName: String):
  case Direct extends WarpShuffleOperator[Int](I32, "__shfl_sync", "sourceLane")
  case Up extends WarpShuffleOperator[UInt](U32, "__shfl_up_sync", "delta")
  case Down extends WarpShuffleOperator[UInt](U32, "__shfl_down_sync", "delta")
  case Xor extends WarpShuffleOperator[Int](I32, "__shfl_xor_sync", "laneMask")
