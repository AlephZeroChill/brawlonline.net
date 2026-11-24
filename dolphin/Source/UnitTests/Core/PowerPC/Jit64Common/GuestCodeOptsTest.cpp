// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "Common/BitSet.h"
#include "Common/CommonTypes.h"
#include "Common/ScopeGuard.h"
#include "Common/x64ABI.h"
#include "Core/Core.h"
#include "Core/PowerPC/ConditionRegister.h"
#include "Core/PowerPC/Gekko.h"
#include "Core/PowerPC/Jit64/Jit.h"
#include "Core/PowerPC/Jit64Common/Jit64AsmCommon.h"
#include "Core/PowerPC/Jit64Common/Jit64Constants.h"
#include "Core/PowerPC/PPCAnalyst.h"
#include "Core/System.h"

// After x64Emitter.h (via Jit.h), which declares XEmitter::TEST.
#include <gtest/gtest.h>

namespace
{
PPCAnalyst::CodeOp MakeDForm(u32 opcd, u32 rsd, u32 ra, s16 simm, BitSet32 regs_out = {})
{
  PPCAnalyst::CodeOp op{};
  op.inst.hex = (opcd << 26) | ((rsd & 31) << 21) | ((ra & 31) << 16) |
                (static_cast<u16>(simm) & 0xFFFFu);
  op.regsOut = regs_out;
  return op;
}

PPCAnalyst::CodeOp MakePsq(u32 opcd, u32 fs, u32 ra, u32 w, u32 i, s16 simm12,
                           BitSet32 regs_out = {})
{
  PPCAnalyst::CodeOp op{};
  op.inst.hex = (opcd << 26) | ((fs & 31) << 21) | ((ra & 31) << 16) | ((w & 1) << 15) |
                ((i & 7) << 12) | (static_cast<u16>(simm12) & 0x0FFFu);
  op.regsOut = regs_out;
  return op;
}

using Kind = Jit64::DirtyMarkPlan::Kind;

TEST(PlanDirtyMarks, CoalescesPrologueUpdatingStoreAndSubsequentStores)
{
  // stwu r1, -0x20(r1); stw r0, 0x24(r1); stw r31, 0x1c(r1); stw r30, 0x18(r1)
  std::vector<PPCAnalyst::CodeOp> ops = {
      MakeDForm(37, 1, 1, -0x20, BitSet32{1}),
      MakeDForm(36, 0, 1, 0x24),
      MakeDForm(36, 31, 1, 0x1c),
      MakeDForm(36, 30, 1, 0x18),
  };
  std::vector<Jit64::DirtyMarkPlan> plan;
  Jit64::PlanDirtyMarks(ops, plan);

  ASSERT_EQ(plan.size(), 4u);
  EXPECT_EQ(plan[0].kind, Kind::Leader);
  EXPECT_EQ(plan[0].base, 1u);
  EXPECT_EQ(plan[0].first, -0x20);
  EXPECT_EQ(plan[0].last, 0x07);
  EXPECT_EQ(plan[1].kind, Kind::Covered);
  EXPECT_EQ(plan[2].kind, Kind::Covered);
  EXPECT_EQ(plan[3].kind, Kind::Covered);
}

TEST(PlanDirtyMarks, CoalescesUpdatingStoreChainAndSplitsAtGranuleWindow)
{
  // 20 consecutive stwu r4, 4(r3) -> bytes [4..83]; first 16 fit in [4..67] (<64), 17th starts new
  std::vector<PPCAnalyst::CodeOp> ops;
  for (int i = 0; i < 20; ++i)
    ops.push_back(MakeDForm(37, 4, 3, 4, BitSet32{3}));

  std::vector<Jit64::DirtyMarkPlan> plan;
  Jit64::PlanDirtyMarks(ops, plan);

  ASSERT_EQ(plan.size(), 20u);
  EXPECT_EQ(plan[0].kind, Kind::Leader);
  EXPECT_EQ(plan[0].base, 3u);
  EXPECT_EQ(plan[0].first, 4);
  EXPECT_EQ(plan[0].last, 67);
  for (size_t i = 1; i < 16; ++i)
    EXPECT_EQ(plan[i].kind, Kind::Covered) << i;

  EXPECT_EQ(plan[16].kind, Kind::Leader);
  EXPECT_EQ(plan[16].base, 3u);
  EXPECT_EQ(plan[16].first, 4);
  EXPECT_EQ(plan[16].last, 19);
  for (size_t i = 17; i < 20; ++i)
    EXPECT_EQ(plan[i].kind, Kind::Covered) << i;
}

TEST(PlanDirtyMarks, CoalescesAcrossAddiSelfAndClosesOnNonSelfAddi)
{
  // stw r4, 0(r3); stw r5, 4(r3); addi r3, r3, 8; stw r6, 0(r3); addi r3, r4, 8; stw r7, 0(r3)
  std::vector<PPCAnalyst::CodeOp> ops = {
      MakeDForm(36, 4, 3, 0),
      MakeDForm(36, 5, 3, 4),
      MakeDForm(14, 3, 3, 8, BitSet32{3}),
      MakeDForm(36, 6, 3, 0),
      MakeDForm(14, 3, 4, 8, BitSet32{3}),
      MakeDForm(36, 7, 3, 0),
  };
  std::vector<Jit64::DirtyMarkPlan> plan;
  Jit64::PlanDirtyMarks(ops, plan);

  ASSERT_EQ(plan.size(), 6u);
  EXPECT_EQ(plan[0].kind, Kind::Leader);
  EXPECT_EQ(plan[0].base, 3u);
  EXPECT_EQ(plan[0].first, 0);
  EXPECT_EQ(plan[0].last, 11);
  EXPECT_EQ(plan[1].kind, Kind::Covered);
  EXPECT_EQ(plan[2].kind, Kind::Self);
  EXPECT_EQ(plan[3].kind, Kind::Covered);
  EXPECT_EQ(plan[4].kind, Kind::Self);
  EXPECT_EQ(plan[5].kind, Kind::Self);
}

TEST(PlanDirtyMarks, CoalescesPsqStAndPsqStu)
{
  // psq_stu f0, 8(r3), W=0, I=0; psq_st f1, 8(r3), W=0, I=0; psq_st f2, 16(r3), W=1, I=0
  std::vector<PPCAnalyst::CodeOp> ops = {
      MakePsq(61, 0, 3, 0, 0, 8, BitSet32{3}),
      MakePsq(60, 1, 3, 0, 0, 8),
      MakePsq(60, 2, 3, 1, 0, 16),
  };
  std::vector<Jit64::DirtyMarkPlan> plan;
  Jit64::PlanDirtyMarks(ops, plan);

  ASSERT_EQ(plan.size(), 3u);
  EXPECT_EQ(plan[0].kind, Kind::Leader);
  EXPECT_EQ(plan[0].base, 3u);
  EXPECT_EQ(plan[0].first, 8);
  EXPECT_EQ(plan[0].last, 27);
  EXPECT_EQ(plan[1].kind, Kind::Covered);
  EXPECT_EQ(plan[2].kind, Kind::Covered);
}

TEST(PlanDirtyMarks, BarrierClosesUpdatingStoreGroup)
{
  std::vector<PPCAnalyst::CodeOp> ops = {
      MakeDForm(37, 1, 1, -16, BitSet32{1}),
      MakeDForm(36, 0, 1, 4),
      MakeDForm(36, 31, 1, 8),
  };
  std::vector<u8> barriers = {0, 0, 1};
  std::vector<Jit64::DirtyMarkPlan> plan;
  Jit64::PlanDirtyMarks(ops, plan, barriers);

  ASSERT_EQ(plan.size(), 3u);
  EXPECT_EQ(plan[0].kind, Kind::Leader);
  EXPECT_EQ(plan[0].first, -16);
  EXPECT_EQ(plan[0].last, -9);
  EXPECT_EQ(plan[1].kind, Kind::Covered);
  EXPECT_EQ(plan[2].kind, Kind::Self);
}

struct FloatCompareOut
{
  u64 cr_internal;
  u8 lazy_fpcc;
};

class FloatCompareHarness : public CommonAsmRoutines
{
public:
  explicit FloatCompareHarness(Core::System& system) : CommonAsmRoutines(jit), jit(system)
  {
    using namespace Gen;
    AllocCodeSpace(8192);
    m_const_pool.Init(AllocChildCodeSpace(4096), 4096);

    fn = reinterpret_cast<void (*)(const double*, const double*, FloatCompareOut*)>(AlignCode16());
    ABI_PushRegistersAndAdjustStack(ABI_ALL_CALLEE_SAVED, 8, 16);
    MOVSD(XMM4, MatR(ABI_PARAM1));  // Ra
    MOVSD(XMM5, MatR(ABI_PARAM2));  // Rb
    MOV(64, R(RBX), R(ABI_PARAM3));

    const OpArg lazy_dst = MDisp(RBX, offsetof(FloatCompareOut, lazy_fpcc));
    EmitBranchlessFloatCompare(XMM5, R(XMM4), MDisp(RBX, offsetof(FloatCompareOut, cr_internal)),
                               &lazy_dst);

    ABI_PopRegistersAndAdjustStack(ABI_ALL_CALLEE_SAVED, 8, 16);
    RET();
  }

  void (*fn)(const double*, const double*, FloatCompareOut*) = nullptr;
  Jit64 jit;
};

TEST(Jit64, BranchlessFloatCompareMatchesAllCases)
{
  Core::DeclareAsCPUThread();
  Common::ScopeGuard cpu_thread_guard([] { Core::UndeclareAsCPUThread(); });
  const FloatCompareHarness harness(Core::System::GetInstance());

  const double qnan = std::numeric_limits<double>::quiet_NaN();
  const double snan = std::numeric_limits<double>::signaling_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::array<double, 10> vals = {
      -inf, -100.5, -1.0, -0.0, 0.0, 1.0, 100.5, inf, qnan, snan,
  };

  for (double a : vals)
  {
    for (double b : vals)
    {
      FloatCompareOut out{};
      harness.fn(&a, &b, &out);

      u8 expected_cr;
      if (std::isnan(a) || std::isnan(b))
        expected_cr = PowerPC::CR_SO;
      else if (a < b)
        expected_cr = PowerPC::CR_LT;
      else if (a > b)
        expected_cr = PowerPC::CR_GT;
      else
        expected_cr = PowerPC::CR_EQ;

      EXPECT_EQ(out.cr_internal, PowerPC::ConditionRegister::PPCToInternal(expected_cr))
          << "a=" << a << " b=" << b;
      EXPECT_EQ(out.lazy_fpcc, expected_cr) << "a=" << a << " b=" << b;
    }
  }
}
}  // namespace
