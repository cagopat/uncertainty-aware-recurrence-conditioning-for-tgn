"""Static documentation checks, not model execution or a browser render test."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / 'README.md', ROOT / 'THIRD_PARTY_NOTICES.md', *sorted((ROOT / 'docs').glob('*.md'))]


class MarkdownFormatting(unittest.TestCase):
    def test_math_uses_github_delimiters_and_balanced_fences(self):
        for path in DOCUMENTS:
            fence = None
            with self.subTest(document=path.name):
                for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                    if line.startswith('```'):
                        fence = line[3:] if fence is None else None
                        continue
                    if fence is not None:
                        continue
                    self.assertFalse(re.search(r'\\[\[\]()]', line), f'{path.name}:{number}: use GitHub math delimiters')
                    self.assertEqual(len(re.findall(r'(?<!\\)\$', line)) % 2, 0,
                                     f'{path.name}:{number}: unbalanced inline math')
                self.assertIsNone(fence, f'{path.name}: unclosed code/math fence')

    def test_protocol_equations_and_weight_table_use_math(self):
        text = (ROOT / 'docs/PROTOCOL.md').read_text(encoding='utf-8')
        self.assertGreaterEqual(text.count('```math\n'), 2)
        # Ordinary Markdown otherwise consumes the escaped literal set braces.
        self.assertIn(r'$`\rho\in\{0,0.5,1\}`$', text)
        for row in ('| Equal |', '| Fitted soft |', '| Fixed period four |'):
            line = next(value for value in text.splitlines() if value.startswith(row))
            self.assertEqual(line.count('|'), 3, 'Math must not break the two-column table')
            self.assertEqual(line.count('$'), 2)


if __name__ == '__main__':
    unittest.main()
