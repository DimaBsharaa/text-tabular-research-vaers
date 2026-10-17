"""Counterexample checks for the read-only source-row auditor."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from research.scripts.vaers import audit_source_rows as audit
from research.scripts.vaers import build_vaers as bv
from research.scripts.vaers.test_build_vaers import write_zip


class SourceRowAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.raw = cls.root / '2023VAERSData.zip'
        write_zip(cls.raw, n=150)
        data, meta = bv.build([cls.raw], 'v7', n_rows=100000)
        cls.csv = bv.write(data, meta, cls.root / 'candidate', 'v7')
        cls.schema = cls.csv.parent / 'schema.json'

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_declared_fields_reconstruct_on_real_format_fixture(self):
        result = audit.audit(self.csv, self.schema, [self.raw])
        self.assertEqual(len(result['source_reconstruction_checks']), 18)
        self.assertTrue(all(c['status'] == 'PASS' for c in result['source_reconstruction_checks']))

    def test_changed_label_and_background_are_detected_even_with_updated_csv_hash(self):
        df = pd.read_csv(self.csv, keep_default_na=False, dtype={'report_id': str})
        df.loc[0, 'serious'] = 1 - int(df.loc[0, 'serious'])
        df.loc[0, 'history'] = 'fabricated background'
        changed = self.root / 'changed.csv'
        df.to_csv(changed, index=False)
        schema = json.loads(self.schema.read_text())
        schema['csv_sha256'] = hashlib.sha256(changed.read_bytes()).hexdigest()
        changed_schema = self.root / 'changed-schema.json'
        changed_schema.write_text(json.dumps(schema))
        result = audit.audit(changed, changed_schema, [self.raw])
        failed = {c['check']: c['mismatches'] for c in result['source_reconstruction_checks'] if c['status'] == 'FAIL'}
        self.assertEqual(failed, {'target_from_raw_seriousness_flags': 1, 'history_reproduced_by_current_mask': 1})

    def test_source_fingerprint_mismatch_stops_reconstruction(self):
        schema = json.loads(self.schema.read_text())
        schema['sources'][0]['sha256'] = '0' * 64
        bad = self.root / 'bad-schema.json'
        bad.write_text(json.dumps(schema))
        with self.assertRaisesRegex(ValueError, 'Raw source member hashes'):
            audit.audit(self.csv, bad, [self.raw])


if __name__ == '__main__':
    unittest.main()
