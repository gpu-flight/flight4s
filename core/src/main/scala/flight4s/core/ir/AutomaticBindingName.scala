package flight4s.core.ir

import java.util.concurrent.atomic.AtomicLong

/** Internal binding keys, never CUDA identifiers or externally visible ABI names. */
private[core] object AutomaticBindingName:
  private val owners = new AtomicLong()
  private val syntax = "\\$flight4s\\$([0-9]+)\\$([0-9]+)\\$([a-z][A-Za-z0-9_]*)".r

  final class Scope:
    private val owner = owners.getAndIncrement()
    private var ordinal = 0L

    def next(label: String): String =
      val result = s"$$flight4s$$$owner$$$ordinal$$$label"
      ordinal += 1
      result

  def isGenerated(name: String): Boolean = name match
    case syntax(owner, ordinal, _) => owner.toLongOption.isDefined && ordinal.toLongOption.isDefined
    case _ => false

  def emissionOrder(name: String): (Long, String) = name match
    case syntax(_, ordinal, label) => (ordinal.toLong, label)
    case _ => throw new IllegalArgumentException("not an automatic binding")

  def label(name: String): String = emissionOrder(name)._2
