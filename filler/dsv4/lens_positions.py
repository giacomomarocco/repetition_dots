"""Shared required prompt-position coverage for lens artifacts."""
ANSWER_PREFIX = ('answer_word', 'answer_colon', 'answer_prompt')
ANSWER_TEXT = ('Answer', ':', ' ')
FIVE_SHOT_PREFIX = ('answer_word', 'answer_colon', 'assistant', 'end_think')
FIVE_SHOT_TEXT = ('Answer', ':', '<｜Assistant｜>', '</think>')


def prompt_position_labels(filler_length, *, legacy=False, suffix='space'):
    if type(filler_length) is not int or filler_length < 0:
        raise ValueError('filler length must be a nonnegative integer')
    if suffix not in ('space', 'five_shot') or (legacy and suffix != 'space'):
        raise ValueError('unknown or incompatible prompt suffix')
    tail = FIVE_SHOT_PREFIX if suffix == 'five_shot' else ANSWER_PREFIX
    return ['last_question', *[f'filler_{i}' for i in range(filler_length)],
            *(('answer_prompt',) if legacy else tail)]


def validate_positions(cell, decode):
    """Verify label order, contiguous absolute positions and actual prefix tokens."""
    ids, positions = cell['input_ids'], cell['positions']
    suffix = cell.get('position_suffix', 'space')
    expected = prompt_position_labels(cell['filler_length'], suffix=suffix)
    if [p['label'] for p in positions] != expected:
        raise ValueError('all required Answer prefix and suffix positions are required in order')
    start = len(ids) - len(expected)
    if start < 0 or [p['absolute_position'] for p in positions] != list(range(start, len(ids))):
        raise ValueError('incorrect absolute prompt positions')
    for pos in positions:
        token = ids[pos['absolute_position']]
        if pos['token_id'] != token or pos['token'] != decode([token]):
            raise ValueError('position token IDs/text disagree with the prompt')
    text = FIVE_SHOT_TEXT if suffix == 'five_shot' else ANSWER_TEXT
    if [p['token'] for p in positions[-len(text):]] != list(text):
        raise ValueError('expected separate Answer, colon and space tokens or declared five-shot suffix')
    if suffix == 'five_shot':
        if any(p['token'].strip() != '.' for p in positions[1:1 + cell['filler_length']]):
            raise ValueError('expected one dot per filler token')
