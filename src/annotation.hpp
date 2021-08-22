#if !defined(KANACHAN_ANNOTATION_HPP_INCLUDE_GUARD)
#define KANACHAN_ANNOTATION_HPP_INCLUDE_GUARD

#include "round_progress.hpp"
#include "player_state.hpp"
#include "mahjongsoul.pb.h"
#include <iosfwd>
#include <limits>
#include <cstdint>


namespace Kanachan{

class Annotation
{
private:
  Annotation(
    Kanachan::PlayerState const &player_state,
    Kanachan::RoundProgress const &round_progress,
    std::uint_fast8_t prev_dapai_seat,
    std::uint_fast8_t prev_dapai,
    std::vector<lq::OptionalOperation> const &action_candidates);

public:
  Annotation(
    Kanachan::PlayerState const &player_state,
    Kanachan::RoundProgress const &round_progress,
    std::uint_fast8_t prev_dapai_seat,
    std::uint_fast8_t prev_dapai,
    std::vector<lq::OptionalOperation> const &action_candidates,
    lq::RecordDealTile const &);

  Annotation(
    Kanachan::PlayerState const &player_state,
    Kanachan::RoundProgress const &round_progress,
    std::vector<lq::OptionalOperation> const &action_candidates,
    lq::RecordDiscardTile const &record);

  Annotation(
    Kanachan::PlayerState const &player_state,
    Kanachan::RoundProgress const &round_progress,
    std::uint_fast8_t prev_dapai_seat,
    std::uint_fast8_t prev_dapai,
    std::vector<lq::OptionalOperation> const &action_candidates,
    lq::RecordChiPengGang const &record);

  Annotation(
    Kanachan::PlayerState const &player_state,
    Kanachan::RoundProgress const &round_progress,
    std::vector<lq::OptionalOperation> const &action_candidates,
    lq::RecordAnGangAddGang const &record);

  Annotation(
    Kanachan::PlayerState const &player_state,
    Kanachan::RoundProgress const &round_progress,
    std::uint_fast8_t prev_dapai_seat,
    std::uint_fast8_t prev_dapai,
    std::vector<lq::OptionalOperation> const &action_candidates,
    lq::HuleInfo const &record);

  Annotation(
    Kanachan::PlayerState const &player_state,
    Kanachan::RoundProgress const &round_progress,
    std::vector<lq::OptionalOperation> const &action_candidates,
    lq::RecordLiuJu const &);

  Annotation(
    Kanachan::PlayerState const &player_state,
    Kanachan::RoundProgress const &round_progress,
    std::uint_fast8_t prev_dapai_seat,
    std::uint_fast8_t prev_dapai,
    std::vector<lq::OptionalOperation> const &action_candidates,
    lq::RecordNoTile const &);

  Annotation(Annotation const &) = default;

  Annotation &operator=(Annotation const &) = delete;

  void printWithRoundResult(
    std::string const &uuid, std::uint_fast8_t i,
    std::array<Kanachan::PlayerState, 4u> const &player_states,
    Kanachan::RoundProgress const &round_progress,
    std::uint_fast8_t round_result,
    std::array<std::int_fast32_t, 4u> const &round_delta_scores,
    std::array<std::uint_fast8_t, 4u> const &round_ranks,
    std::ostream &os) const;

private:
  static constexpr std::uint_fast16_t dapai_offset_      =   0u;
  static constexpr std::uint_fast16_t angang_offset_     = 148u;
  static constexpr std::uint_fast16_t jiagang_offset_    = 182u;
  static constexpr std::uint_fast16_t zimohu_offset_     = 219u;
  static constexpr std::uint_fast16_t liuju_offset_      = 220u;
  static constexpr std::uint_fast16_t skip_offset_       = 221u;
  static constexpr std::uint_fast16_t chi_offset_        = 222u;
  static constexpr std::uint_fast16_t peng_offset_       = 312u;
  static constexpr std::uint_fast16_t daminggang_offset_ = 423u;
  static constexpr std::uint_fast16_t rong_offset_       = 534u;

  Kanachan::PlayerState player_state_;
  std::uint_fast8_t round_progress_size_;

  // Action
  std::vector<uint_fast16_t> action_candidates_;
  std::uint_fast8_t action_index_ = std::numeric_limits<std::uint_fast8_t>::max();
}; // class Annotation

} // namespace Kanachan

#endif // !defined(KANACHAN_ANNOTATION_HPP_INCLUDE_GUARD)
