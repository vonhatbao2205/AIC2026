import type { Channel } from "../api/types";
import { CHANNEL_LABEL } from "../lib/constants";

export function ChannelBadge({ channel }: { channel: Channel }) {
  return (
    <span className={`badge ${channel}`} title={`matched on ${channel}`}>
      {CHANNEL_LABEL[channel]}
    </span>
  );
}

export function ChannelBadges({ channels }: { channels: Channel[] }) {
  return (
    <span style={{ display: "inline-flex", gap: 3 }}>
      {channels.map((c) => (
        <ChannelBadge key={c} channel={c} />
      ))}
    </span>
  );
}
