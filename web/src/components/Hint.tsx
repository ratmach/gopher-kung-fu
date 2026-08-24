type Props = {
  text: string;
};

export default function Hint({ text }: Props) {
  return (
    <span className="hint">
      <button type="button" className="hint-q" aria-label="What is this?">
        ?
      </button>
      <span className="hint-tip" role="tooltip">
        {text}
      </span>
    </span>
  );
}
