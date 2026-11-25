// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "Common/CommonTypes.h"
#include "Common/Swap.h"
#include "VideoCommon/LookUpTables.h"
#include "VideoCommon/TextureDecoder.h"

namespace
{
u32 ReferenceRGB5A3(u16 val)
{
  int r, g, b, a;
  if (val & 0x8000)
  {
    r = Convert5To8((val >> 10) & 0x1f);
    g = Convert5To8((val >> 5) & 0x1f);
    b = Convert5To8(val & 0x1f);
    a = 0xFF;
  }
  else
  {
    a = Convert3To8((val >> 12) & 0x7);
    r = Convert4To8((val >> 8) & 0xf);
    g = Convert4To8((val >> 4) & 0xf);
    b = Convert4To8(val & 0xf);
  }
  return static_cast<u32>(r | (g << 8) | (b << 16) | (a << 24));
}

void ReferenceDecodeXFB(u8* dst, const u8* src, u32 width, u32 height, u32 stride)
{
  const u8* src_ptr = src;
  u8* dst_ptr = dst;
  for (u32 y = 0; y < height; ++y)
  {
    const u8* row_ptr = src_ptr;
    for (u32 x = 0; x < width; x += 2)
    {
      const int Y1 = int(*(row_ptr++)) - 16;
      const int U = int(*(row_ptr++)) - 128;
      const int Y2 = int(*(row_ptr++)) - 16;
      const int V = int(*(row_ptr++)) - 128;

      const u8 R1 = static_cast<u8>(std::clamp(int(1.164f * Y1 + 1.596f * V), 0, 255));
      const u8 G1 = static_cast<u8>(std::clamp(int(1.164f * Y1 - 0.392f * U - 0.813f * V), 0, 255));
      const u8 B1 = static_cast<u8>(std::clamp(int(1.164f * Y1 + 2.017f * U), 0, 255));

      const u8 R2 = static_cast<u8>(std::clamp(int(1.164f * Y2 + 1.596f * V), 0, 255));
      const u8 G2 = static_cast<u8>(std::clamp(int(1.164f * Y2 - 0.392f * U - 0.813f * V), 0, 255));
      const u8 B2 = static_cast<u8>(std::clamp(int(1.164f * Y2 + 2.017f * U), 0, 255));

      u32 rgba = 0xff000000 | B1 << 16 | G1 << 8 | R1;
      std::memcpy(dst_ptr, &rgba, sizeof(rgba));
      dst_ptr += sizeof(rgba);
      rgba = 0xff000000 | B2 << 16 | G2 << 8 | R2;
      std::memcpy(dst_ptr, &rgba, sizeof(rgba));
      dst_ptr += sizeof(rgba);
    }
    src_ptr += stride;
  }
}
}  // namespace

// Covers all 65,536 16-bit RGB5A3 pixel values in a 256x256 image (4096 4x4 tiles). With
// mixed_rows, bit 15 alternates inside every 4-pixel row (the per-pixel select path); without,
// each row is all RGB555 or all RGBA4443 (the two uniform paths).
static void CheckRGB5A3Exhaustive(bool mixed_rows)
{
  constexpr int width = 256;
  constexpr int height = 256;
  constexpr int total_pixels = width * height;

  std::vector<u16> tiled_src_be(total_pixels);
  std::vector<u32> expected(total_pixels);

  for (int tile_y = 0; tile_y < height / 4; ++tile_y)
  {
    for (int tile_x = 0; tile_x < width / 4; ++tile_x)
    {
      const int tile_idx = tile_y * (width / 4) + tile_x;
      for (int iy = 0; iy < 4; ++iy)
      {
        for (int ix = 0; ix < 4; ++ix)
        {
          const int linear_idx = tile_idx * 16 + iy * 4 + ix;
          // XOR bit 15 with (ix & 1) for mixed rows; still a bijection on u16.
          const u16 val = static_cast<u16>(linear_idx ^ (mixed_rows ? (ix & 1) << 15 : 0));
          tiled_src_be[linear_idx] = Common::swap16(val);
          const int py = tile_y * 4 + iy;
          const int px = tile_x * 4 + ix;
          expected[py * width + px] = ReferenceRGB5A3(val);
        }
      }
    }
  }

  std::vector<u32> actual(total_pixels, 0);
  TexDecoder_Decode(reinterpret_cast<u8*>(actual.data()),
                    reinterpret_cast<const u8*>(tiled_src_be.data()), width, height,
                    TextureFormat::RGB5A3, nullptr, TLUTFormat::IA8);

  ASSERT_EQ(actual, expected);
}

TEST(TextureDecoderTest, RGB5A3MixedRows)
{
  CheckRGB5A3Exhaustive(true);
}

TEST(TextureDecoderTest, RGB5A3UniformRows)
{
  CheckRGB5A3Exhaustive(false);
}

TEST(TextureDecoderTest, XFBDecodeExactness)
{
  // Test both 4-pixel-aligned and non-4-pixel-aligned widths (tail path) with padded stride,
  // covering extreme clamping (0 and 255) as well as mid-range Y/U/V values.
  for (const u32 width : {2u, 4u, 6u, 16u, 38u, 640u})
  {
    const u32 height = 8;
    const u32 stride = width * 2 + 16;  // extra padding between rows
    std::vector<u8> src(height * stride, 0xA5);

    u32 seed = 0x12345678u ^ width;
    auto next_byte = [&]() -> u8 {
      seed = seed * 1664525u + 1013904223u;
      return static_cast<u8>(seed >> 24);
    };

    for (u32 y = 0; y < height; ++y)
    {
      for (u32 x = 0; x < width * 2; ++x)
      {
        if (y == 0)
          src[y * stride + x] = (x & 1) ? 0 : 0;  // extreme low clamp
        else if (y == 1)
          src[y * stride + x] = 255;  // extreme high clamp
        else
          src[y * stride + x] = next_byte();
      }
    }

    std::vector<u8> expected(width * height * 4, 0xCC);
    std::vector<u8> actual(width * height * 4, 0x33);

    ReferenceDecodeXFB(expected.data(), src.data(), width, height, stride);
    TexDecoder_DecodeXFB(actual.data(), src.data(), width, height, stride);

    ASSERT_EQ(actual, expected) << "Mismatch for width=" << width;
  }
}

TEST(TextureDecoderTest, XFBDecodeAllYUV)
{
  // Every (Y, U, V) combination: one 256-pixel row per (U, V) pair, Y running 0..255.
  constexpr u32 width = 256;
  constexpr u32 height = 256 * 256;
  constexpr u32 stride = width * 2;
  std::vector<u8> src(size_t(height) * stride);
  for (u32 uv = 0; uv < height; ++uv)
  {
    u8* row = &src[size_t(uv) * stride];
    for (u32 x = 0; x < width; x += 2)
    {
      row[x * 2 + 0] = static_cast<u8>(x);
      row[x * 2 + 1] = static_cast<u8>(uv & 0xff);
      row[x * 2 + 2] = static_cast<u8>(x + 1);
      row[x * 2 + 3] = static_cast<u8>(uv >> 8);
    }
  }
  std::vector<u8> expected(size_t(width) * height * 4);
  std::vector<u8> actual(size_t(width) * height * 4);
  ReferenceDecodeXFB(expected.data(), src.data(), width, height, stride);
  TexDecoder_DecodeXFB(actual.data(), src.data(), width, height, stride);
  ASSERT_EQ(actual, expected);
}
