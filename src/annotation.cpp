#include "annotation.hpp"

#include "round_progress.hpp"
#include "player_state.hpp"
#include "utility.hpp"
#include "throw.hpp"
#include "mahjongsoul.pb.h"
#include <regex>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>


namespace Kanachan{

namespace{

using std::placeholders::_1;

} // namespace *unnamed*

Annotation::Annotation(
  Kanachan::PlayerState const &player_state,
  Kanachan::RoundProgress const &round_progress,
  std::uint_fast8_t const prev_dapai_seat,
  std::uint_fast8_t const prev_dapai,
  std::vector<lq::OptionalOperation> const &action_candidates)
  : player_state_(player_state),
    round_progress_size_(round_progress.getSize()),
    action_candidates_()
{
  bool skippable = false;
  bool discardable = false;
  bool skippable_in_liqi = false;

  for (auto const &action_candidate : action_candidates) {
    switch (action_candidate.type()) {
    case 1u:
    {
      // 打牌
      if (prev_dapai_seat != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      auto discardable_tiles = player_state_.getDiscardableTiles();
      std::vector<std::uint_fast8_t> forbidden_tiles;
      for (auto const &forbidden_tile : action_candidate.combination()) {
        forbidden_tiles.push_back(Kanachan::pai2Num(forbidden_tile));
      }
      for (std::uint_fast8_t const tile : discardable_tiles) {
        {
          bool forbidden = false;
          for (std::uint_fast8_t const forbidden_tile : forbidden_tiles) {
            if (tile == forbidden_tile) {
              forbidden = true;
              break;
            }
          }
          if (forbidden) {
            continue;
          }
        }
        std::uint_fast16_t const moqie = 0u;
        std::uint_fast16_t const liqi = 0u;
        std::uint_fast16_t const action_code = dapai_offset_ + 2u * 2u * tile + 2u * moqie + liqi;
        action_candidates_.push_back(action_code);
      }

      std::uint_fast16_t const zimopai = player_state_.getZimopai();
      if (zimopai != std::numeric_limits<std::uint_fast8_t>::max()) {
        std::uint_fast16_t const moqie = 1u;
        std::uint_fast16_t const liqi = 0u;
        std::uint_fast16_t const action_code
          = dapai_offset_ + 2u * 2u * zimopai + 2u * moqie + liqi;
        action_candidates_.push_back(action_code);
      }
      discardable = true;
      break;
    }
    case 2u:
    {
      // チー
      if (prev_dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai_seat == player_state_.getSeat()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      for (auto const &tiles : action_candidate.combination()) {
        std::regex r("(..)\\|(..)");
        std::smatch m;
        if (!std::regex_match(tiles, m, r)) {
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }
        std::uint_fast8_t const tile0 = Kanachan::pai2Num(m[1u]);
        std::uint_fast8_t const tile1 = Kanachan::pai2Num(m[2u]);
        std::uint_fast16_t const action_code
          = chi_offset_ + Kanachan::encodeChi(tile0, tile1, prev_dapai);
        action_candidates_.push_back(action_code);
      }
      skippable = true;
      break;
    }
    case 3u:
    {
      // ポン
      if (prev_dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai_seat == player_state_.getSeat()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      std::uint_fast8_t const from
        = (4u + prev_dapai_seat - player_state_.getSeat()) % 4u - 1u;
      for (auto const &tiles : action_candidate.combination()) {
        std::uint_fast16_t const action_code = peng_offset_ + 37 * from + prev_dapai;
        action_candidates_.push_back(action_code);
      }
      skippable = true;
      break;
    }
    case 4u:
    {
      // 暗槓
      for (auto const &tiles : action_candidate.combination()) {
        std::regex r("(..)\\|(..)\\|(..)\\|(..)");
        std::smatch m;
        if (!std::regex_match(tiles, m, r)) {
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }
        std::uint_fast8_t tile = Kanachan::pai2Num(m[4u]);
        if (tile == 0u || tile == 10u || tile == 20u) {
          tile += 5u;
        }
        if (tile < 10u) {
          tile -= 1u;
        }
        else if (tile < 20u) {
          tile -= 2u;
        }
        else {
          tile -= 3u;
        }
        std::uint_fast16_t const action_code = angang_offset_ + tile;
        action_candidates_.push_back(action_code);
      }
      skippable_in_liqi = true;
      break;
    }
    case 5u:
    {
      // 大明槓
      if (prev_dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai_seat == player_state_.getSeat()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      std::uint_fast8_t const from
        = (4u + prev_dapai_seat - player_state_.getSeat()) % 4u - 1u;
      for (auto const &tiles : action_candidate.combination()) {
        std::uint_fast16_t const action_code
          = daminggang_offset_ + 37 * from + prev_dapai;
        action_candidates_.push_back(action_code);
      }
      skippable = true;
      break;
    }
    case 6u:
    {
      // 加槓
      for (auto const &tiles : action_candidate.combination()) {
        std::regex r("(..)\\|(..)\\|(..)\\|(..)");
        std::smatch m;
        if (!std::regex_match(tiles, m, r)) {
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }
        std::uint_fast8_t tile = Kanachan::pai2Num(m[4u]);
        {
          bool found = false;
          for (std::uint_fast8_t const discardable_tile : player_state_.getDiscardableTiles()) {
            if (tile == discardable_tile) {
              found = true;
              break;
            }
            if ((tile == 5u || tile == 15u || tile == 25u) && tile - 5u == discardable_tile) {
              tile -= 5u;
              found = true;
              break;
            }
          }
          if (!found) {
            if (tile == player_state_.getZimopai()) {
              found = true;
            }
            else if ((tile == 5u || tile == 15u || tile == 25u) && tile - 5u == player_state_.getZimopai()) {
              tile -= 5u;
              found = true;
            }
          }
          if (!found) {
            KANACHAN_THROW<std::logic_error>("A logic error.");
          }
        }
        std::uint_fast16_t const action_code = jiagang_offset_ + tile;
        action_candidates_.push_back(action_code);
      }
      skippable_in_liqi = true;
      break;
    }
    case 7u:
    {
      // 立直
      if (prev_dapai_seat != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      for (std::string const &pai : action_candidate.combination()) {
        // API のバグ？ 赤を切って立直できるときに黒が combination に乗っていない．
        std::array<std::uint_fast8_t, 2u> tiles = {
          Kanachan::pai2Num(pai), std::numeric_limits<std::uint_fast8_t>::max()
        };
        if (tiles[0u] == 0u || tiles[0u] == 10u || tiles[0u] == 20u) {
          tiles[1u] = tiles[0u] + 5u;
        }
        auto const discardable_tiles = player_state_.getDiscardableTiles();
        std::array<bool, 2u> found = {false, false};
        for (std::uint_fast8_t const discardable_tile : discardable_tiles) {
          for (std::uint_fast8_t i = 0u; i < tiles.size(); ++i) {
            std::uint_fast8_t const tile = tiles[i];
            if (discardable_tile == tile) {
              found[i] = true;
            }
          }
        }
        for (std::uint_fast8_t i = 0u; i < found.size(); ++i) {
          if (found[i]) {
            std::uint_fast16_t const tile = tiles[i];
            std::uint_fast16_t const moqie = 0u;
            std::uint_fast16_t const liqi = 1u;
            std::uint_fast16_t const action_code = dapai_offset_ + 2u * 2u * tile + 2u * moqie + liqi;
            action_candidates_.push_back(action_code);
          }
        }
        for (std::uint_fast8_t i = 0u; i < tiles.size(); ++i) {
          std::uint_fast16_t const tile = tiles[i];
          if (player_state_.getZimopai() == tile) {
            std::uint_fast16_t const moqie = 1u;
            std::uint_fast16_t const liqi = 1u;
            std::uint_fast16_t const action_code
              = dapai_offset_ + 2u * 2u * tile + 2u * moqie + liqi;
            action_candidates_.push_back(action_code);
            found[i] = true;
          }
        }
        if (!found[0u] && !found[1u]) {
          KANACHAN_THROW<std::logic_error>("A logic error.");
        }
      }
      break;
    }
    case 8u:
    {
      // 自摸和
      if (prev_dapai_seat != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      std::uint_fast16_t const action_code = zimohu_offset_;
      action_candidates_.push_back(action_code);
      skippable_in_liqi = true;
      break;
    }
    case 9u:
    {
      // ロン
      if (prev_dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai_seat == player_state_.getSeat()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai == std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      std::uint_fast16_t const from
        = (4u + prev_dapai_seat - player_state_.getSeat()) % 4u - 1u;
      std::uint_fast16_t const action_code = rong_offset_ + from;
      action_candidates_.push_back(action_code);
      skippable = true;
      break;
    }
    case 10u:
    {
      // 九種九牌
      if (prev_dapai_seat != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      if (prev_dapai != std::numeric_limits<std::uint_fast8_t>::max()) {
        KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
      }
      std::uint_fast16_t const action_code = liuju_offset_;
      action_candidates_.push_back(action_code);
      break;
    }
    default:
      KANACHAN_THROW<std::logic_error>(_1)
        << action_candidate.type() << ": an unknown type.";
      break;
    }
  }

  if (skippable) {
    std::uint_fast16_t const action_code = skip_offset_;
    action_candidates_.push_back(action_code);
  }

  if (!discardable && skippable_in_liqi) {
    // 立直中に暗槓・加槓・自摸和が可能な場合に自摸切りの選択肢を加える．
    if (player_state_.getZimopai() == std::numeric_limits<std::uint_fast8_t>::max()) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    std::uint_fast16_t const tile = player_state_.getZimopai();
    std::uint_fast16_t const moqie = 1u;
    std::uint_fast16_t const liqi = 0u;
    std::uint_fast16_t const action_code
      = dapai_offset_ + 2u * 2u * tile + 2u * moqie + liqi;
    action_candidates_.push_back(action_code);
  }

  std::sort(action_candidates_.begin(), action_candidates_.end());
  auto last = std::unique(action_candidates_.begin(), action_candidates_.end());
  action_candidates_.erase(last, action_candidates_.end());
}

Annotation::Annotation(
  Kanachan::PlayerState const &player_state,
  Kanachan::RoundProgress const &round_progress,
  std::uint_fast8_t prev_dapai_seat,
  std::uint_fast8_t prev_dapai,
  std::vector<lq::OptionalOperation> const &action_candidates,
  lq::RecordDealTile const &)
  : Annotation(
      player_state, round_progress, prev_dapai_seat, prev_dapai,
      action_candidates)
{
  std::uint_fast16_t const action = skip_offset_;
  if (action >= chi_offset_) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }

  action_index_ = std::ranges::find(action_candidates_, action) - action_candidates_.cbegin();
  if (action_index_ >= action_candidates_.size()) {
    std::ostringstream oss;
    if (action_candidates_.size() > 0u) {
      oss << action_candidates_[0u];
      for (std::uint_fast8_t i = 1u; i < action_candidates_.size(); ++i) {
        oss << ',' << action_candidates_[i];
      }
    }
    KANACHAN_THROW<std::logic_error>(_1) << oss.str() << '\t' << action;
  }
}

Annotation::Annotation(
  Kanachan::PlayerState const &player_state,
  Kanachan::RoundProgress const &round_progress,
  std::vector<lq::OptionalOperation> const &action_candidates,
  lq::RecordDiscardTile const &record)
  : Annotation(
      player_state, round_progress,
      std::numeric_limits<std::uint_fast8_t>::max(),
      std::numeric_limits<std::uint_fast8_t>::max(),
      action_candidates)
{
  std::uint_fast16_t const tile = Kanachan::pai2Num(record.tile());
  std::uint_fast16_t const moqie = record.moqie() ? 1 : 0;
  std::uint_fast16_t const liqi = record.is_liqi() || record.is_wliqi() ? 1 : 0;
  std::uint_fast16_t const action = dapai_offset_ + 2u * 2u * tile + 2u * moqie + liqi;
  if (action >= angang_offset_) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }

  action_index_ = std::ranges::find(action_candidates_, action) - action_candidates_.cbegin();
  if (action_index_ >= action_candidates_.size() && player_state_.getLeftTileCount() == 69u) {
    // 親の配牌時自摸切り（実際には親の配牌に手牌と自摸牌の区別は無いので便宜上の処理）
    std::uint_fast16_t const tile = Kanachan::pai2Num(record.tile());
    std::uint_fast16_t const moqie = 1;
    std::uint_fast16_t const liqi = record.is_liqi() || record.is_wliqi() ? 1 : 0;
    std::uint_fast16_t const action = dapai_offset_ + 2u * 2u * tile + 2u * moqie + liqi;
    if (action >= angang_offset_) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    action_index_ = std::ranges::find(action_candidates_, action) - action_candidates_.cbegin();
  }
  if (action_index_ >= action_candidates_.size()) {
    std::ostringstream oss;
    if (action_candidates_.size() > 0u) {
      oss << action_candidates_[0u];
      for (std::uint_fast8_t i = 1u; i < action_candidates_.size(); ++i) {
        oss << ',' << action_candidates_[i];
      }
    }
    KANACHAN_THROW<std::logic_error>(_1) << oss.str() << '\t' << action;
  }
}

Annotation::Annotation(
  Kanachan::PlayerState const &player_state,
  Kanachan::RoundProgress const &round_progress,
  std::uint_fast8_t const prev_dapai_seat,
  std::uint_fast8_t const prev_dapai,
  std::vector<lq::OptionalOperation> const &action_candidates,
  lq::RecordChiPengGang const &record)
  : Annotation(
      player_state, round_progress, prev_dapai_seat, prev_dapai,
      action_candidates)
{
  std::uint_fast16_t action = std::numeric_limits<std::uint_fast16_t>::max();

  if (record.type() == 0) {
    // チー
    std::array<std::uint_fast8_t, 3u> tiles{
      Kanachan::pai2Num(record.tiles()[0u]),
      Kanachan::pai2Num(record.tiles()[1u]),
      Kanachan::pai2Num(record.tiles()[2u])
    };
    std::uint_fast16_t const encode = Kanachan::encodeChi(tiles.cbegin(), tiles.cend());
    action = chi_offset_ + encode;
    if (action >= peng_offset_) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
  }
  else if (record.type() == 1u) {
    // ポン
    std::uint_fast16_t const from = record.froms()[2u];
    std::uint_fast16_t const relative = (4u + from - record.seat()) % 4u - 1u;
    std::uint_fast16_t const tile = Kanachan::pai2Num(record.tiles()[2u]);
    action = peng_offset_ + 37u * relative + tile;
    if (action >= daminggang_offset_) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
  }
  else if (record.type() == 2u) {
    // 大明槓
    std::uint_fast16_t const from = record.froms()[3u];
    std::uint_fast16_t const relative = (4u + from - record.seat()) % 4u - 1u;
    std::uint_fast16_t const tile = Kanachan::pai2Num(record.tiles()[3u]);
    action = daminggang_offset_ + 37u * relative + tile;
    if (action >= rong_offset_) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
  }
  else {
    KANACHAN_THROW<std::runtime_error>(_1) << "A broken data: type = " << record.type();
  }

  action_index_ = std::ranges::find(action_candidates_, action) - action_candidates_.cbegin();
  if (action_index_ >= action_candidates_.size()) {
    std::ostringstream oss;
    if (action_candidates_.size() > 0u) {
      oss << action_candidates_[0u];
      for (std::uint_fast8_t i = 1u; i < action_candidates_.size(); ++i) {
        oss << ',' << action_candidates_[i];
      }
    }
    KANACHAN_THROW<std::logic_error>(_1) << oss.str() << '\t' << action;
  }
}

Annotation::Annotation(
  Kanachan::PlayerState const &player_state,
  Kanachan::RoundProgress const &round_progress,
  std::vector<lq::OptionalOperation> const &action_candidates,
  lq::RecordAnGangAddGang const &record)
  : Annotation(
      player_state, round_progress,
      std::numeric_limits<std::uint_fast8_t>::max(),
      std::numeric_limits<std::uint_fast8_t>::max(), action_candidates)
{
  std::uint_fast16_t action = std::numeric_limits<std::uint_fast16_t>::max();

  if (record.type() == 2u) {
    // 加槓
    std::uint_fast16_t const tile = Kanachan::pai2Num(record.tiles());
    action = jiagang_offset_ + tile;
    if (action >= zimohu_offset_) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
  }
  else if (record.type() == 3u) {
    // 暗槓
    std::uint_fast16_t tile = Kanachan::pai2Num(record.tiles());
    if (tile == 0u || tile == 10u || tile == 20u) {
      tile += 5u;
    }
    if (tile < 10u) {
      tile -= 1u;
    }
    else if (tile < 20u) {
      tile -= 2u;
    }
    else {
      tile -= 3u;
    }
    action = angang_offset_ + tile;
    if (action >= jiagang_offset_) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
  }
  else {
    KANACHAN_THROW<std::runtime_error>(_1) << "A broken data: type = " << record.type();
  }

  action_index_ = std::ranges::find(action_candidates_, action) - action_candidates_.cbegin();
  if (action_index_ >= action_candidates_.size()) {
    std::ostringstream oss;
    if (action_candidates_.size() > 0u) {
      oss << action_candidates_[0u];
      for (std::uint_fast8_t i = 1u; i < action_candidates_.size(); ++i) {
        oss << ',' << action_candidates_[i];
      }
    }
    KANACHAN_THROW<std::logic_error>(_1) << oss.str() << '\t' << action;
  }
}

Annotation::Annotation(
  Kanachan::PlayerState const &player_state,
  Kanachan::RoundProgress const &round_progress,
  std::uint_fast8_t const prev_dapai_seat,
  std::uint_fast8_t const prev_dapai,
  std::vector<lq::OptionalOperation> const &action_candidates,
  lq::HuleInfo const &record)
  : Annotation(
      player_state, round_progress, prev_dapai_seat, prev_dapai,
      action_candidates)
{
  std::uint_fast16_t action = std::numeric_limits<std::uint_fast16_t>::max();

  if (prev_dapai_seat == std::numeric_limits<std::uint_fast8_t>::max()) {
    // 自摸和
    action = zimohu_offset_;
    if (action >= liuju_offset_) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
  }
  else {
    // 栄和
    if (prev_dapai_seat == player_state_.getSeat()) {
      KANACHAN_THROW<std::logic_error>("A logic error.");
    }
    std::uint_fast16_t const from
      = (4u + prev_dapai_seat - player_state_.getSeat()) % 4u - 1u;
    action = rong_offset_ + from;
  }

  action_index_ = std::ranges::find(action_candidates_, action) - action_candidates_.cbegin();
  if (action_index_ >= action_candidates_.size()) {
    std::ostringstream oss;
    if (action_candidates_.size() > 0u) {
      oss << action_candidates_[0u];
      for (std::uint_fast8_t i = 1u; i < action_candidates_.size(); ++i) {
        oss << ',' << action_candidates_[i];
      }
    }
    KANACHAN_THROW<std::logic_error>(_1) << oss.str() << '\t' << action;
  }
}

Annotation::Annotation(
  Kanachan::PlayerState const &player_state,
  Kanachan::RoundProgress const &round_progress,
  std::vector<lq::OptionalOperation> const &action_candidates,
  lq::RecordLiuJu const &)
  : Annotation(
      player_state, round_progress,
      std::numeric_limits<std::uint_fast8_t>::max(),
      std::numeric_limits<std::uint_fast8_t>::max(), action_candidates)
{
  std::uint_fast16_t const action = liuju_offset_;
  if (action >= skip_offset_) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }

  action_index_ = std::ranges::find(action_candidates_, action) - action_candidates_.cbegin();
  if (action_index_ >= action_candidates_.size()) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
}

Annotation::Annotation(
  Kanachan::PlayerState const &player_state,
  Kanachan::RoundProgress const &round_progress,
  std::uint_fast8_t prev_dapai_seat,
  std::uint_fast8_t prev_dapai,
  std::vector<lq::OptionalOperation> const &action_candidates,
  lq::RecordNoTile const &)
  : Annotation(
      player_state, round_progress, prev_dapai_seat, prev_dapai,
      action_candidates)
{
  std::uint_fast16_t const action = skip_offset_;
  if (action >= chi_offset_) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }

  action_index_ = std::ranges::find(action_candidates_, action) - action_candidates_.cbegin();
  if (action_index_ >= action_candidates_.size()) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
}

void Annotation::printWithRoundResult(
  std::string const &uuid, std::uint_fast8_t i,
  std::array<Kanachan::PlayerState, 4u> const &player_states,
  Kanachan::RoundProgress const &round_progress,
  std::uint_fast8_t const round_result,
  std::array<std::int_fast32_t, 4u> const &round_delta_scores,
  std::array<std::uint_fast8_t, 4u> const &round_ranks, std::ostream &os) const
{
  if (round_result >= 19u) {
    KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
  }
  for (auto const &round_rank : round_ranks) {
    if (round_rank >= 4u) {
      KANACHAN_THROW<std::invalid_argument>("An invalid argument.");
    }
  }

  os << uuid;
  os << '\t';
  player_state_.print(player_states, os);
  os << '\t';
  round_progress.print(round_progress_size_, os);
  os << '\t';
  if (action_candidates_.empty()) {
    KANACHAN_THROW<std::logic_error>("A logic error.");
  }
  os << action_candidates_[0u];
  for (std::uint_fast8_t i = 1u; i < action_candidates_.size(); ++i) {
    os << ',' << action_candidates_[i];
  }
  os << '\t';
  os << static_cast<unsigned>(action_index_);
  os << '\t';
  os << static_cast<unsigned>(round_result)
     << ',' << round_delta_scores[i]
     << ',' << round_delta_scores[(i + 1u) % 4u]
     << ',' << round_delta_scores[(i + 2u) % 4u]
     << ',' << round_delta_scores[(i + 3u) % 4u]
     << ',' << static_cast<unsigned>(round_ranks[i])
     << ',' << static_cast<unsigned>(round_ranks[(i + 1u) % 4u])
     << ',' << static_cast<unsigned>(round_ranks[(i + 2u) % 4u])
     << ',' << static_cast<unsigned>(round_ranks[(i + 3u) % 4u])
     << ',' << static_cast<unsigned>(player_state_.getGameRank())
     << ',' << player_state_.getGameScore()
     << ',' << player_state_.getDeltaGradingPoint()
     << '\n';
}

} // namespace Kanachan
