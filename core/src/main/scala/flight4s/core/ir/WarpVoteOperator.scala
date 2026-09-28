package flight4s.core.ir

import flight4s.core.types.{Bool, CudaType, U32, UInt}

enum WarpVoteOperator[T](val resultType: CudaType[T], val cudaName: String):
  case Ballot extends WarpVoteOperator[UInt](U32, "__ballot_sync")
  case All extends WarpVoteOperator[Boolean](Bool, "__all_sync")
  case Any extends WarpVoteOperator[Boolean](Bool, "__any_sync")
