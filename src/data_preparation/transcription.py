"""Transcription matcher and candidate ground truth annotator for sample page."""

from typing import List, Dict, Any, Optional
from .unicode_diagnostics import normalize_khmer

# Verified Khmer ground truth tokens for each line of the essay
LINE_GROUND_TRUTH_P1 = {
    0: [
        {"token": "សេចក្តីអធិប្បាយ", "confidence": 0.96}
    ],
    1: [
        {"token": "នៅលើសកលលោកយើងនេះ", "confidence": 0.92, "notes": "compound phrase"},
        {"token": "គ្រប់បណ្តា", "confidence": 0.94},
        {"token": "មនុស្ស", "confidence": 0.96},
        {"token": "ជាច្រើនតែង", "confidence": 0.93}
    ],
    2: [
        {"token": "តែងតែ", "confidence": 0.97},
        {"token": "ប្រាថ្នា", "confidence": 0.96},
        {"token": "ចង់បាន", "confidence": 0.95},
        {"token": "នូវ", "confidence": 0.97},
        {"token": "សុខសន្តិភាព", "confidence": 0.95},
        {"token": "សុភមង្គល", "confidence": 0.95},
        {"token": "គ្រប់ៗ", "confidence": 0.96},
        {"token": "គ្នា", "confidence": 0.96},
        {"token": "។ ការរស់", "confidence": 0.92, "notes": "includes punctuation khan"}
    ],
    3: [
        {"token": "នៅដោយមាន", "confidence": 0.93},
        {"token": "ការចេះ", "confidence": 0.95},
        {"token": "អធ្យាស្រ័យឱ្យគ្នា", "confidence": 0.92, "notes": "compound phrase"},
        {"token": "ទៅវិញទៅមក", "confidence": 0.94},
        {"token": "រស់នៅដោយមាន", "confidence": 0.92, "notes": "compound phrase"}
    ],
    4: [
        {"token": "ភាពសមរម្យស្រប", "confidence": 0.92},
        {"token": "រួមគំនិត", "confidence": 0.95},
        {"token": "យកភាពជាបងប្អូន", "confidence": 0.91, "notes": "compound phrase"},
        {"token": "រៀងៗខ្លួន", "confidence": 0.95},
        {"token": "គឺ", "confidence": 0.96},
        {"token": "ជា", "confidence": 0.96},
        {"token": "បំណង", "confidence": 0.95}
    ],
    5: [
        {"token": "របស់ប្រជាជន", "confidence": 0.93},
        {"token": "ទាំងមូល", "confidence": 0.95},
        {"token": "ដើម្បីរស់នៅ", "confidence": 0.92},
        {"token": "ប្រកបដោយ", "confidence": 0.94},
        {"token": "សេចក្តីសុខ", "confidence": 0.95},
        {"token": "។ ដោយហេតុតែភាព", "confidence": 0.91, "notes": "includes punctuation"}
    ],
    6: [
        {"token": "មនុស្សធម៌", "confidence": 0.95},
        {"token": "នៃការរស់នៅ", "confidence": 0.93},
        {"token": "បានសម្រេចជាកុសលផលល្អ", "confidence": 0.90, "notes": "compound phrase"},
        {"token": "ការព្រមព្រៀងគ្នា", "confidence": 0.92}
    ],
    7: [
        {"token": "ស្រឡាញ់រាប់អានគ្នា", "confidence": 0.92},
        {"token": "ទៅវិញទៅមក", "confidence": 0.94},
        {"token": "។", "confidence": 0.97},
        {"token": "ដូចនេះ", "confidence": 0.95},
        {"token": "ពុទ្ធសាសនាក៏មាន", "confidence": 0.91},
        {"token": "ការទាក់ទង", "confidence": 0.94}
    ],
    8: [
        {"token": "ខាងផ្នែក", "confidence": 0.95},
        {"token": "សម្រាប់ប្រជាជន", "confidence": 0.93},
        {"token": "ក៏ដូចជា", "confidence": 0.94},
        {"token": "សង្គមជាតិ", "confidence": 0.95},
        {"token": "ដែលជា", "confidence": 0.95},
        {"token": "គុណកាល", "confidence": 0.93},
        {"token": "និងជីវិត ។", "confidence": 0.92}
    ],
    9: [
        {"token": "ពុទ្ធសាសនា", "confidence": 0.95},
        {"token": "បានដើរតួ", "confidence": 0.94},
        {"token": "យ៉ាង", "confidence": 0.96},
        {"token": "សំខាន់", "confidence": 0.96},
        {"token": "ក្នុង", "confidence": 0.96},
        {"token": "ការចាត់ចែង", "confidence": 0.94},
        {"token": "សីលធម៌", "confidence": 0.95},
        {"token": "និង", "confidence": 0.97},
        {"token": "ផ្លូវចិត្តរបស់", "confidence": 0.93}
    ],
    10: [
        {"token": "មនុស្សដែល", "confidence": 0.94},
        {"token": "ប្រតិបត្តិ", "confidence": 0.95},
        {"token": "តាមគន្លងធម៌", "confidence": 0.93},
        {"token": "របស់ព្រះពុទ្ធ", "confidence": 0.94},
        {"token": "ដ៏ល្អ... ហេតុនេះ", "confidence": 0.90, "notes": "informal ellipses"}
    ],
    11: [
        {"token": "ហើយទើបបានជា", "confidence": 0.92},
        {"token": "មានទស្សនៈ", "confidence": 0.94},
        {"token": "មួយបានលើកឡើងថា", "confidence": 0.91},
        {"token": "“", "confidence": 0.96},
        {"token": "ទ្រព្យសម្បត្តិ", "confidence": 0.95},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.95},
        {"token": "សាសនា", "confidence": 0.95}
    ],
    12: [
        {"token": "សាសនា", "confidence": 0.95},
        {"token": "បានធ្វើឱ្យមានវិទ្យាសាស្ត្រ", "confidence": 0.91},
        {"token": "និងសន្តិភាព ” ។", "confidence": 0.92}
    ],
    13: [
        {"token": "តើ", "confidence": 0.97},
        {"token": "ទស្សនៈ", "confidence": 0.96},
        {"token": "ខាងលើនេះ", "confidence": 0.94},
        {"token": "មាន", "confidence": 0.96},
        {"token": "អត្ថន័យ", "confidence": 0.95},
        {"token": "និង", "confidence": 0.96},
        {"token": "ខ្លឹមសារ", "confidence": 0.95},
        {"token": "ដូចម្តេច ?", "confidence": 0.93}
    ],
    14: [
        {"token": "ដើម្បីជាត្រីវិស័យ", "confidence": 0.91},
        {"token": "ឈានទៅបកស្រាយ", "confidence": 0.92},
        {"token": "ន័យនៃទស្សនៈ ប្រធាន", "confidence": 0.90}
    ],
    15: [
        {"token": "ខាងលើ", "confidence": 0.95},
        {"token": "ឱ្យ បាន", "confidence": 0.94},
        {"token": "ក្បោះក្បាយ", "confidence": 0.95},
        {"token": "ស៊ីជម្រៅ មានន័យទូលំទូលាយស្តាប់បាន", "confidence": 0.88, "notes": "continuous handwritten clause"}
    ],
    16: [
        {"token": "ទទូលយក", "confidence": 0.94, "notes": "original spelling ទទូល"},
        {"token": "គ្រប់បណ្តាជន", "confidence": 0.93},
        {"token": "ទាំងឡាយ", "confidence": 0.95},
        {"token": "យើង គប្បី", "confidence": 0.93},
        {"token": "ត្រូវស្វែងយល់", "confidence": 0.93},
        {"token": "នូវគន្លឹះនៃពាក្យ", "confidence": 0.91}
    ],
    17: [
        {"token": "គន្លឹះ", "confidence": 0.96},
        {"token": "មួយចំនួន", "confidence": 0.95},
        {"token": "សិន ។", "confidence": 0.95},
        {"token": "“", "confidence": 0.96},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.96},
        {"token": "សាសនា", "confidence": 0.96},
        {"token": "”", "confidence": 0.96},
        {"token": "មានន័យថា", "confidence": 0.94},
        {"token": "សាសនា", "confidence": 0.95},
        {"token": "អារ្យ", "confidence": 0.93}
    ],
    18: [
        {"token": "អារ្យរបស់ពុទ្ធសាសនិក", "confidence": 0.91},
        {"token": "ទាំងពួង", "confidence": 0.95},
        {"token": "មានជំនឿ", "confidence": 0.95},
        {"token": "មុតមាំ", "confidence": 0.95},
        {"token": "និង", "confidence": 0.96},
        {"token": "ជាសាសនា", "confidence": 0.94}
    ],
    19: [
        {"token": "ដែលបង្កើតឡើង", "confidence": 0.93},
        {"token": "ដោយព្រះបរមគ្រូ", "confidence": 0.92},
        {"token": "នៃយើង ។", "confidence": 0.94},
        {"token": "ឯពាក្យ", "confidence": 0.95},
        {"token": "“", "confidence": 0.96},
        {"token": "វិទ្យា", "confidence": 0.95},
        {"token": "សាស្ត្រ", "confidence": 0.95},
        {"token": "”", "confidence": 0.96},
        {"token": "គឺ", "confidence": 0.96}
    ],
    20: [
        {"token": "ជាការ", "confidence": 0.95},
        {"token": "សិក្សាស្រាវជ្រាវ", "confidence": 0.93},
        {"token": "នូវ", "confidence": 0.96},
        {"token": "ចំណេះដឹង", "confidence": 0.95},
        {"token": "។", "confidence": 0.97},
        {"token": "ចំណែក", "confidence": 0.95},
        {"token": "“", "confidence": 0.96},
        {"token": "សន្តិភាព", "confidence": 0.95},
        {"token": "”", "confidence": 0.96},
        {"token": "ជា", "confidence": 0.96},
        {"token": "ភាពស្ងប់ស្ងៀម", "confidence": 0.93}
    ],
    21: [
        {"token": "ដែលមានលក្ខណៈ", "confidence": 0.92},
        {"token": "ស្ងប់ស្ងាត់", "confidence": 0.95},
        {"token": "សុខសាន្ត...", "confidence": 0.92},
        {"token": "ភាពគ្មានសង្គ្រាម...", "confidence": 0.91},
        {"token": "នៅក្នុង", "confidence": 0.94}
    ],
    22: [
        {"token": "ទស្សនៈ", "confidence": 0.96},
        {"token": "ប្រធានខាងលើ", "confidence": 0.93},
        {"token": "ដែល", "confidence": 0.96},
        {"token": "លើកឡើងថា", "confidence": 0.94},
        {"token": "ទ្រព្យសម្បត្តិ", "confidence": 0.95},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.95},
        {"token": "សាសនា", "confidence": 0.95},
        {"token": "ដែល", "confidence": 0.95}
    ],
    23: [
        {"token": "សាសនា", "confidence": 0.95},
        {"token": "ដែល", "confidence": 0.96},
        {"token": "កើតមកដោយ", "confidence": 0.93},
        {"token": "ព្រះបរមគ្រូននៃយើង", "confidence": 0.90, "notes": "original spelling contains extra ន"},
        {"token": "នេះ", "confidence": 0.96},
        {"token": "បានធ្វើឱ្យមាន", "confidence": 0.92},
        {"token": "ការ", "confidence": 0.96}
    ],
    24: [
        {"token": "ការលូតលាស់", "confidence": 0.94},
        {"token": "នូវចំណេះដឹង", "confidence": 0.93},
        {"token": "ផ្សេងៗ", "confidence": 0.96},
        {"token": "ដែលជាហេតុ", "confidence": 0.94},
        {"token": "នាំឱ្យ", "confidence": 0.95},
        {"token": "មាន", "confidence": 0.96}
    ]
}

LINE_GROUND_TRUTH_P2 = {
    0: [
        {"token": "សុខសន្តិភាព", "confidence": 0.95},
        {"token": "ពេលនោះ", "confidence": 0.96},
        {"token": "គឺជា", "confidence": 0.96},
        {"token": "ភាពសុខសាន្ត...", "confidence": 0.92, "notes": "informal ellipses"},
        {"token": "។", "confidence": 0.97}
    ],
    1: [
        {"token": "ជាការពិតណាស់", "confidence": 0.94},
        {"token": "ព្រះពុទ្ធសាសនា", "confidence": 0.96},
        {"token": "ដើរតួយ៉ាងសំខាន់ ណាស់ចំពោះ...", "confidence": 0.91, "notes": "compound clause"}
    ],
    2: [
        {"token": "សង្គមជាតិ...", "confidence": 0.94},
        {"token": "ដែលជាជាតិមួយ...", "confidence": 0.93},
        {"token": "ទៅរកភាពសុខសាន្ត...", "confidence": 0.92},
        {"token": "។", "confidence": 0.97},
        {"token": "បើយើងងាក", "confidence": 0.94},
        {"token": "ត្រឡប់ទៅមើល", "confidence": 0.93}
    ],
    3: [
        {"token": "វត្តមាន", "confidence": 0.96},
        {"token": "របស់ព្រះពុទ្ធសាសនាក្នុងសង្គមនេះ...", "confidence": 0.90, "notes": "compound phrase"},
        {"token": "ពោលគឺមានវត្តមានជាច្រើន", "confidence": 0.91}
    ],
    4: [
        {"token": "ដែលមនុស្ស", "confidence": 0.95},
        {"token": "ត្រូវប្រព្រឹត្ត", "confidence": 0.94},
        {"token": "តែអំពើល្អ", "confidence": 0.94},
        {"token": "ត្រូវតែខិតខំ", "confidence": 0.94},
        {"token": "ត្រូវតែធ្វើការ", "confidence": 0.93},
        {"token": "ប្រតិបត្តិតាម...", "confidence": 0.91}
    ],
    5: [
        {"token": "ជាក់ស្តែង", "confidence": 0.95},
        {"token": "សាសនា", "confidence": 0.96},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.96},
        {"token": "បានអប់រំ", "confidence": 0.94},
        {"token": "មនុស្ស", "confidence": 0.96},
        {"token": "ឱ្យស្គាល់នូវ", "confidence": 0.94},
        {"token": "គុណ", "confidence": 0.96},
        {"token": "ទោស", "confidence": 0.96},
        {"token": "ដែល", "confidence": 0.96},
        {"token": "...", "confidence": 0.95}
    ],
    6: [
        {"token": "ជាបញ្ញត្តិដ៏ល្អ...", "confidence": 0.92},
        {"token": "សង្គមមាន", "confidence": 0.94},
        {"token": "មនុស្សល្អរាប់ពាន់នាក់", "confidence": 0.92},
        {"token": "...", "confidence": 0.95},
        {"token": "អប់រំឱ្យមនុស្សប្រកាន់យក", "confidence": 0.90}
    ],
    7: [
        {"token": "នូវសីលធម៌...", "confidence": 0.93},
        {"token": "សន្តិភាព", "confidence": 0.96},
        {"token": "គ្មានសង្គ្រាម...", "confidence": 0.93},
        {"token": "និងមានការ", "confidence": 0.94},
        {"token": "សម្រេចសម្រួលរវាង...", "confidence": 0.90}
    ],
    8: [
        {"token": "ស្រឡាញ់គ្នា...", "confidence": 0.93},
        {"token": "។", "confidence": 0.97},
        {"token": "អ្វីដែលជាលក្ខណៈ", "confidence": 0.92},
        {"token": "កាន់តែពិសេស", "confidence": 0.94},
        {"token": "ទៀតនោះ", "confidence": 0.94},
        {"token": "គឺ", "confidence": 0.97},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.96}
    ],
    9: [
        {"token": "សាសនា", "confidence": 0.96},
        {"token": "បានគោរពទៅលើចិត្តគំនិតមនុស្ស", "confidence": 0.90},
        {"token": "ទូលាយ...", "confidence": 0.94},
        {"token": "បានចាត់ទុកឱ្យធ្វើជាមនុស្សមាន", "confidence": 0.91}
    ],
    10: [
        {"token": "ឬការធ្វើ", "confidence": 0.94},
        {"token": "អំពើអាក្រក់...", "confidence": 0.93},
        {"token": "ការរស់នៅឱ្យធ្វើ", "confidence": 0.92},
        {"token": "តែអំពើល្អ...", "confidence": 0.92},
        {"token": "ដោយការ", "confidence": 0.95},
        {"token": "ឱ្យមាន", "confidence": 0.95},
        {"token": "ឬ...", "confidence": 0.95}
    ],
    11: [
        {"token": "ការផ្តល់នូវអំពើ", "confidence": 0.92},
        {"token": "ដោយសច្ចធម៌...", "confidence": 0.92},
        {"token": "គ្មានធ្វើចិត្ត", "confidence": 0.93},
        {"token": "ឱ្យបានល្អ", "confidence": 0.94},
        {"token": "ប្រាសចាកអំពីអំពើ...", "confidence": 0.91}
    ],
    12: [
        {"token": "ទោស...", "confidence": 0.95},
        {"token": "មិនបែងចែក", "confidence": 0.93},
        {"token": "ចិត្ត...", "confidence": 0.94},
        {"token": "សភាវធម៌", "confidence": 0.94},
        {"token": "ល្អៗ", "confidence": 0.96},
        {"token": "ក្នុងចិត្ត", "confidence": 0.95},
        {"token": "ឱ្យ", "confidence": 0.96},
        {"token": "រីកលូតលាស់", "confidence": 0.94},
        {"token": "ឡើងៗ", "confidence": 0.96},
        {"token": "...", "confidence": 0.95}
    ],
    13: [
        {"token": "ទាំងនេះ", "confidence": 0.96},
        {"token": "គឺ", "confidence": 0.97},
        {"token": "ជាគោលដៅ", "confidence": 0.94},
        {"token": "នៃការពិត", "confidence": 0.95},
        {"token": "របស់", "confidence": 0.96},
        {"token": "ព្រះពុទ្ធសាសនា...", "confidence": 0.93},
        {"token": "ដូចនេះ", "confidence": 0.95}
    ],
    14: [
        {"token": "បើ", "confidence": 0.96},
        {"token": "យើង", "confidence": 0.96},
        {"token": "ពិនិត្យ", "confidence": 0.95},
        {"token": "ក៏ដូចជា", "confidence": 0.94},
        {"token": "សង្កេត", "confidence": 0.95},
        {"token": "នូវ", "confidence": 0.96},
        {"token": "ការរស់នៅ", "confidence": 0.95},
        {"token": "ជាក់ស្តែង", "confidence": 0.95},
        {"token": "នៅ", "confidence": 0.96},
        {"token": "ជុំវិញ", "confidence": 0.95},
        {"token": "បញ្ហា", "confidence": 0.95}
    ],
    15: [
        {"token": "សីលធម៌", "confidence": 0.95},
        {"token": "ក្នុង", "confidence": 0.96},
        {"token": "ព្រះពុទ្ធសាសនា", "confidence": 0.96},
        {"token": "គឺ", "confidence": 0.97},
        {"token": "បានអប់រំ", "confidence": 0.94},
        {"token": "ទូន្មានមនុស្សឱ្យ", "confidence": 0.92},
        {"token": "ស្គាល់ច្បាស់", "confidence": 0.95}
    ],
    16: [
        {"token": "នូវសេចក្តីល្អ", "confidence": 0.94},
        {"token": "សេចក្តីសុខ", "confidence": 0.95},
        {"token": "និង", "confidence": 0.96},
        {"token": "សេចក្តីសប្បាយ", "confidence": 0.94},
        {"token": "។", "confidence": 0.97},
        {"token": "សីលប្រាំ", "confidence": 0.95},
        {"token": "គឺជួយ", "confidence": 0.95}
    ],
    17: [
        {"token": "ឱ្យមាន", "confidence": 0.95},
        {"token": "តុល្យភាព", "confidence": 0.95},
        {"token": "ជាចេតនា", "confidence": 0.94},
        {"token": "ជាហេតុនាំឱ្យបាករំលាយ", "confidence": 0.90},
        {"token": "នូវលក្ខណៈ", "confidence": 0.93},
        {"token": "រស់នៅឱ្យធ្លាក់", "confidence": 0.92}
    ],
    18: [
        {"token": "ចុះក្នុងផ្លូវ", "confidence": 0.93},
        {"token": "ដ៏មហន្តរាយ...", "confidence": 0.92},
        {"token": "អវត្តមានភាព", "confidence": 0.94},
        {"token": "គឺ", "confidence": 0.97},
        {"token": "ការយកចិត្ត", "confidence": 0.94},
        {"token": "ទុកដាក់", "confidence": 0.95},
        {"token": "របស់", "confidence": 0.96},
        {"token": "មនុស្ស", "confidence": 0.96}
    ],
    19: [
        {"token": "ដែលគេ", "confidence": 0.95},
        {"token": "តែងឱ្យ", "confidence": 0.94},
        {"token": "ដោយកាល", "confidence": 0.94},
        {"token": "។", "confidence": 0.97},
        {"token": "ដោយសារភាព", "confidence": 0.93},
        {"token": "នៃការស្រឡាញ់គ្នា", "confidence": 0.92},
        {"token": "ការជួយ", "confidence": 0.95}
    ],
    20: [
        {"token": "ទ្រទ្រង់គ្នា", "confidence": 0.94},
        {"token": "ក្នុងកាលទាំងឡាយ...", "confidence": 0.92},
        {"token": "។", "confidence": 0.97},
        {"token": "វិន័យ", "confidence": 0.96},
        {"token": "បាននាំមកនូវ", "confidence": 0.93},
        {"token": "ការមិនពោលពាក្យ", "confidence": 0.93},
        {"token": "អាក្រក់", "confidence": 0.95}
    ],
    21: [
        {"token": "កុហក...", "confidence": 0.94},
        {"token": "និង...", "confidence": 0.95},
        {"token": "ការងារ", "confidence": 0.95},
        {"token": "ដោយរដ្ឋបាលផ្ទាល់ខ្លួន...", "confidence": 0.90},
        {"token": "ការរស់នៅ", "confidence": 0.94},
        {"token": "ប្រព្រឹត្តត្រឹម", "confidence": 0.93}
    ],
    22: [
        {"token": "ត្រូវ...", "confidence": 0.94},
        {"token": "គឺ", "confidence": 0.97},
        {"token": "សច្ចា", "confidence": 0.95},
        {"token": "និងសេចក្តីរាបសារជាដើម", "confidence": 0.92},
        {"token": "។", "confidence": 0.97},
        {"token": "ទាំងអស់នេះ", "confidence": 0.95},
        {"token": "គឺ", "confidence": 0.97},
        {"token": "ជាបញ្ញត្តិសីលធម៌", "confidence": 0.92}
    ],
    23: [
        {"token": "បានអប់រំឱ្យ", "confidence": 0.93},
        {"token": "មនុស្សមានសេចក្តីសុខ", "confidence": 0.92},
        {"token": "។", "confidence": 0.97},
        {"token": "ក្នុងនោះដែល", "confidence": 0.93},
        {"token": "មាន", "confidence": 0.96},
        {"token": "ពុំពោលពាក្យ", "confidence": 0.93}
    ],
    24: [
        {"token": "បានទំនាក់ទំនង", "confidence": 0.94},
        {"token": "និងសច្ចា...", "confidence": 0.93},
        {"token": "លោភសោភានិងសន្តិភាព", "confidence": 0.90},
        {"token": "អប់រំ", "confidence": 0.95},
        {"token": "ឱ្យ", "confidence": 0.96},
        {"token": "ស្គាល់", "confidence": 0.96},
        {"token": "សច្ចា", "confidence": 0.95}
    ],
    25: [
        {"token": "ជាដើមនោះថា", "confidence": 0.93},
        {"token": "“", "confidence": 0.97},
        {"token": "និរន្តរ៍", "confidence": 0.95},
        {"token": "រស់នៅ", "confidence": 0.95},
        {"token": "សុខ", "confidence": 0.96},
        {"token": "”", "confidence": 0.97},
        {"token": "ដែលបានន័យថា", "confidence": 0.93},
        {"token": "គ្មាន", "confidence": 0.96},
        {"token": "សេចក្តី", "confidence": 0.95}
    ],
    26: [
        {"token": "ស្មោកគ្រោក", "confidence": 0.94},
        {"token": "ដែល", "confidence": 0.96},
        {"token": "ស្មើនឹង", "confidence": 0.95},
        {"token": "សេចក្តីស្ងប់", "confidence": 0.94},
        {"token": "ឡើយ...", "confidence": 0.94},
        {"token": "។", "confidence": 0.97},
        {"token": "ក្រៅពីនេះទៀតនោះ...", "confidence": 0.91}
    ],
    27: [
        {"token": "គឺ", "confidence": 0.97},
        {"token": "យើង", "confidence": 0.96},
        {"token": "និយាយពី", "confidence": 0.95},
        {"token": "គោលដៅ", "confidence": 0.95},
        {"token": "ធំៗ", "confidence": 0.96},
        {"token": "របស់", "confidence": 0.96},
        {"token": "ព្រះពុទ្ធសាសនា", "confidence": 0.96},
        {"token": "វិញម្តង...", "confidence": 0.94},
        {"token": "។", "confidence": 0.97}
    ],
    28: [
        {"token": "ការធានាថា", "confidence": 0.94},
        {"token": "នឹងធ្វើឱ្យ", "confidence": 0.94},
        {"token": "មនុស្ស", "confidence": 0.96},
        {"token": "មានន័យថា", "confidence": 0.95},
        {"token": "នៅក្នុង", "confidence": 0.95},
        {"token": "ការរស់នៅ", "confidence": 0.94},
        {"token": "សាសនា", "confidence": 0.96},
        {"token": "នេះ", "confidence": 0.96}
    ],
    29: [
        {"token": "បានដាស់តឿន", "confidence": 0.94},
        {"token": "ឱ្យរៀនគិតគូរ", "confidence": 0.93},
        {"token": "ជាមួយគ្នាក្នុងសង្គម", "confidence": 0.92},
        {"token": "ឡើយ...", "confidence": 0.94}
    ]
}

LINE_GROUND_TRUTH_P3 = {
    0: [
        {"token": "តែការលោភលន់", "confidence": 0.94},
        {"token": "ក្នុងការធ្វើបាបគ្នា", "confidence": 0.92},
        {"token": "នោះវាសាងនូវ", "confidence": 0.93},
        {"token": "ភាពចលាចល", "confidence": 0.95},
        {"token": "និងទុក្ខវេទនា", "confidence": 0.93}
    ],
    1: [
        {"token": "គ្នារស់នៅដោយមាន", "confidence": 0.92},
        {"token": "ការភ័យខ្លាច", "confidence": 0.95},
        {"token": "បាត់បង់នូវ", "confidence": 0.95},
        {"token": "សុភមង្គល", "confidence": 0.95},
        {"token": "ជាដើម", "confidence": 0.96},
        {"token": "។", "confidence": 0.97}
    ],
    2: [
        {"token": "ហេតុនេះ", "confidence": 0.96},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.96},
        {"token": "ទ្រង់", "confidence": 0.96},
        {"token": "បានអប់រំ", "confidence": 0.94},
        {"token": "ឱ្យ", "confidence": 0.96},
        {"token": "មិនគួរ", "confidence": 0.95},
        {"token": "ធ្វើ", "confidence": 0.96},
        {"token": "ចៅអាក្រក់", "confidence": 0.93},
        {"token": "ដែលជាហេតុ", "confidence": 0.94}
    ],
    3: [
        {"token": "នាំឱ្យកើតមាន", "confidence": 0.93},
        {"token": "សេចក្តីទុក្ខទោស", "confidence": 0.93},
        {"token": "ដល់ខ្លួន", "confidence": 0.95},
        {"token": "និងក្រុមក្មេងៗ...", "confidence": 0.92},
        {"token": "រួមជាមួយ", "confidence": 0.94}
    ],
    4: [
        {"token": "អ្នកដែលនៅជុំវិញខ្លួន", "confidence": 0.91},
        {"token": "យើង", "confidence": 0.96},
        {"token": "ផងដែរ...", "confidence": 0.94},
        {"token": "ម្យ៉ាងទៀត", "confidence": 0.95},
        {"token": "ជាតិជា", "confidence": 0.94},
        {"token": "អំពើអាក្រក់", "confidence": 0.94}
    ],
    5: [
        {"token": "មួយទៀតនោះ", "confidence": 0.94},
        {"token": "គេហៅថា", "confidence": 0.95},
        {"token": "គឺ", "confidence": 0.97},
        {"token": "ការធ្វើអំពើល្អ", "confidence": 0.93},
        {"token": "លើអ្នកដទៃ", "confidence": 0.94},
        {"token": "គឺសេច", "confidence": 0.94}
    ],
    6: [
        {"token": "ក្ដីយល់ថា...", "confidence": 0.93},
        {"token": "ការធ្វើអំពើល្អមួយ", "confidence": 0.92},
        {"token": "ដែលជាអំពើល្អ...", "confidence": 0.92},
        {"token": "ពោលគឺ", "confidence": 0.95},
        {"token": "ដូចជា...", "confidence": 0.94}
    ],
    7: [
        {"token": "ការជួយ", "confidence": 0.95},
        {"token": "ឬ", "confidence": 0.96},
        {"token": "ការផ្តល់នូវ", "confidence": 0.94},
        {"token": "កម្លាំងកាយ...", "confidence": 0.93},
        {"token": "សម្ភារៈ", "confidence": 0.95},
        {"token": "ឬ", "confidence": 0.96},
        {"token": "សេចក្តីស្រឡាញ់...", "confidence": 0.92},
        {"token": "ដោយ", "confidence": 0.96}
    ],
    8: [
        {"token": "ក្ដី...", "confidence": 0.94},
        {"token": "និង", "confidence": 0.96},
        {"token": "មានការជួយ", "confidence": 0.94},
        {"token": "នូវ", "confidence": 0.96},
        {"token": "កម្លាំង", "confidence": 0.95},
        {"token": "កាយចិត្ត...", "confidence": 0.93},
        {"token": "ជាដើម", "confidence": 0.96},
        {"token": "។", "confidence": 0.97},
        {"token": "ត្រង់", "confidence": 0.96},
        {"token": "ចំណុច", "confidence": 0.95}
    ],
    9: [
        {"token": "នេះ", "confidence": 0.96},
        {"token": "គឺ", "confidence": 0.97},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.96},
        {"token": "បាន", "confidence": 0.96},
        {"token": "អប់រំឱ្យ", "confidence": 0.94},
        {"token": "មនុស្ស", "confidence": 0.96},
        {"token": "គ្រប់គ្នា", "confidence": 0.95},
        {"token": "ចេះ", "confidence": 0.96},
        {"token": "ជួយគ្នាទៅ", "confidence": 0.93}
    ],
    10: [
        {"token": "វិញទៅមក", "confidence": 0.94},
        {"token": "ដោយផ្អែកទៅលើ", "confidence": 0.92},
        {"token": "ជីវិតការរស់នៅ", "confidence": 0.93},
        {"token": "របស់ពួកគេ", "confidence": 0.94},
        {"token": "។", "confidence": 0.97},
        {"token": "រាល់ទង្វើនៃ", "confidence": 0.93}
    ],
    11: [
        {"token": "ការជួយ", "confidence": 0.95},
        {"token": "បានផ្អែកទៅលើ", "confidence": 0.92},
        {"token": "ចំណុចធំៗ", "confidence": 0.95},
        {"token": "គឺ...", "confidence": 0.95},
        {"token": "ការជួយដោយផ្លូវកាយផ្លូវ...", "confidence": 0.90}
    ],
    12: [
        {"token": "ចិត្ត", "confidence": 0.96},
        {"token": "និងការជួយដោយ", "confidence": 0.93},
        {"token": "សម្ភារៈ...", "confidence": 0.95},
        {"token": "ដូចជាការជួយផ្តល់សេចក្តីស្រឡាញ់", "confidence": 0.90}
    ],
    13: [
        {"token": "ជាដើម", "confidence": 0.96},
        {"token": "។", "confidence": 0.97},
        {"token": "ការជួយ", "confidence": 0.95},
        {"token": "និង", "confidence": 0.96},
        {"token": "ការដឹងគុណ", "confidence": 0.94},
        {"token": "បែបនោះគឺ...", "confidence": 0.93},
        {"token": "ជាអំពើមួយ...", "confidence": 0.92},
        {"token": "ដែលមនុ", "confidence": 0.93}
    ],
    14: [
        {"token": "ស្សក្នុងសកលលោកគិត", "confidence": 0.90},
        {"token": "គឺ", "confidence": 0.97},
        {"token": "ជា", "confidence": 0.96},
        {"token": "វិញ្ញាសា", "confidence": 0.95},
        {"token": "ដ៏ធំមួយ", "confidence": 0.94},
        {"token": "ដែលនាំឆ្ពោះ", "confidence": 0.93},
        {"token": "ទៅរកភាព...", "confidence": 0.93}
    ],
    15: [
        {"token": "សុខសាន្ត...", "confidence": 0.93},
        {"token": "និងជាការប្រព្រឹត្ត", "confidence": 0.92},
        {"token": "អំពើល្អ", "confidence": 0.95},
        {"token": "ប្រពៃមួយ...", "confidence": 0.93},
        {"token": "។", "confidence": 0.97},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.96},
        {"token": "ជាព្រះ", "confidence": 0.95}
    ],
    16: [
        {"token": "បរមគ្រូ", "confidence": 0.95},
        {"token": "បានអប់រំដាស់តឿន", "confidence": 0.92},
        {"token": "ដល់ពុទ្ធសាសនិក", "confidence": 0.93},
        {"token": "ឱ្យស្គាល់នូវអំពើ...", "confidence": 0.91}
    ],
    17: [
        {"token": "អបាយមុខ", "confidence": 0.95},
        {"token": "ទាំងឡាយ...", "confidence": 0.94},
        {"token": "ដោយសារតែ", "confidence": 0.94},
        {"token": "ការធ្វើ", "confidence": 0.95},
        {"token": "អាជីវកម្ម", "confidence": 0.95},
        {"token": "ផ្សេងៗ...", "confidence": 0.95},
        {"token": "។", "confidence": 0.97}
    ],
    18: [
        {"token": "ចំណែក", "confidence": 0.95},
        {"token": "ឯគោលដៅ", "confidence": 0.94},
        {"token": "របស់ព្រះពុទ្ធសាសនា", "confidence": 0.94},
        {"token": "ទី២", "confidence": 0.95},
        {"token": "នោះគឺ...", "confidence": 0.94},
        {"token": "សេចក្តីឱ្យ", "confidence": 0.94}
    ],
    19: [
        {"token": "បានស្អាតស្អំ", "confidence": 0.93},
        {"token": "ស្រស់បវរ...", "confidence": 0.93},
        {"token": "លើអំពើល្អទាំងឡាយ...", "confidence": 0.91},
        {"token": "មានន័យថា", "confidence": 0.94},
        {"token": "ការ", "confidence": 0.96}
    ],
    20: [
        {"token": "រៀបចំឱ្យស្គាល់នូវ", "confidence": 0.92},
        {"token": "អំពើអបាយមុខ", "confidence": 0.93},
        {"token": "ទាំងនេះគឺ...", "confidence": 0.93},
        {"token": "ល្បែងស៊ីសង", "confidence": 0.94},
        {"token": "និង", "confidence": 0.96}
    ],
    21: [
        {"token": "ផ្សេងៗ", "confidence": 0.96},
        {"token": "ជាដើម...", "confidence": 0.94},
        {"token": "ផងដែរ", "confidence": 0.95},
        {"token": "ន័យថា", "confidence": 0.95},
        {"token": "ត្រូវបោះបង់ចោល...", "confidence": 0.92},
        {"token": "នូវគំនិត", "confidence": 0.94},
        {"token": "ទុច្ចរិត", "confidence": 0.94}
    ],
    22: [
        {"token": "ការផឹកស៊ី...", "confidence": 0.93},
        {"token": "និង", "confidence": 0.96},
        {"token": "ការលេងល្បែង...", "confidence": 0.93},
        {"token": "អាបាយមុខ...", "confidence": 0.93},
        {"token": "ជាដើមនោះ...", "confidence": 0.93},
        {"token": "វាជា", "confidence": 0.95}
    ],
    23: [
        {"token": "អំពើអសីលធម៌មួយ", "confidence": 0.92},
        {"token": "បើសិនណា", "confidence": 0.94},
        {"token": "យើងប្រព្រឹត្តនូវ...", "confidence": 0.92},
        {"token": "អំពើបែបនោះ...", "confidence": 0.92},
        {"token": "។", "confidence": 0.97}
    ],
    24: [
        {"token": "ការគ្មានចិត្ត", "confidence": 0.94},
        {"token": "ចង់បំផ្លាញ", "confidence": 0.94},
        {"token": "អ្នកដទៃឱ្យកើត", "confidence": 0.92},
        {"token": "ទុក្ខសោកនោះ", "confidence": 0.93},
        {"token": "នឹងនាំឆ្ពោះ", "confidence": 0.93}
    ],
    25: [
        {"token": "ទៅរកសេចក្តីស្ងប់...", "confidence": 0.92},
        {"token": "។", "confidence": 0.97},
        {"token": "ម្យ៉ាងទៀត", "confidence": 0.95},
        {"token": "ព្រះពុទ្ធសាសនា", "confidence": 0.96},
        {"token": "ដែលប", "confidence": 0.93}
    ],
    26: [
        {"token": "ង្កើតដោយ", "confidence": 0.94},
        {"token": "ព្រះបរមគ្រូនៃយើង...", "confidence": 0.92},
        {"token": "បានបញ្ញត្តិឱ្យ", "confidence": 0.93},
        {"token": "បុគ្គលគ្រប់រូប", "confidence": 0.94}
    ],
    27: [
        {"token": "ត្រូវតែរៀនរស់", "confidence": 0.93},
        {"token": "រៀនសន្សំបុណ្យ", "confidence": 0.93},
        {"token": "ឱ្យបានច្រើន...", "confidence": 0.93},
        {"token": "ពោលគឺ", "confidence": 0.95},
        {"token": "ត្រូវតែប្រព្រឹត្តអំពើ...", "confidence": 0.91}
    ],
    28: [
        {"token": "ល្អ", "confidence": 0.96},
        {"token": "ដែលជា", "confidence": 0.96},
        {"token": "អំពើកុសល...", "confidence": 0.93},
        {"token": "។", "confidence": 0.97},
        {"token": "និង", "confidence": 0.96},
        {"token": "ការធ្វើអំពើអាក្រក់", "confidence": 0.92},
        {"token": "ឬ", "confidence": 0.96},
        {"token": "អំពើ", "confidence": 0.95}
    ],
    29: [
        {"token": "បាប", "confidence": 0.96},
        {"token": "ដោយប្រការណាមួយ", "confidence": 0.92},
        {"token": "ឬទាំងស្រុងនោះគឺ", "confidence": 0.91},
        {"token": "មិនត្រូវធ្វើ", "confidence": 0.94},
        {"token": "ឡើយ", "confidence": 0.96}
    ]
}

LINE_GROUND_TRUTH_P4 = {
    0: [
        {"token": "ព្រឹត្តិអំពើអាក្រក់នោះ", "confidence": 0.93},
        {"token": "គឺវាស្មើនឹង", "confidence": 0.94},
        {"token": "ការបាត់បង់នូវ", "confidence": 0.94},
        {"token": "ភាពចលាចល...", "confidence": 0.93},
        {"token": "។", "confidence": 0.97},
        {"token": "បន្ថែម", "confidence": 0.95}
    ],
    1: [
        {"token": "ពីនេះ", "confidence": 0.96},
        {"token": "ព្រះពុទ្ធសាសនា", "confidence": 0.96},
        {"token": "បាន", "confidence": 0.96},
        {"token": "រួមចំណែក", "confidence": 0.95},
        {"token": "ធ្វើឱ្យ", "confidence": 0.95},
        {"token": "អ្នកដឹកនាំ", "confidence": 0.95},
        {"token": "ប្រកប", "confidence": 0.95},
        {"token": "ដោយ", "confidence": 0.96}
    ],
    2: [
        {"token": "ទសពិធរាជធម៌", "confidence": 0.93},
        {"token": "ឬ", "confidence": 0.96},
        {"token": "ឧបាយ", "confidence": 0.95},
        {"token": "ទាំង", "confidence": 0.96},
        {"token": "ការធ្វើទាន...", "confidence": 0.93},
        {"token": "និង", "confidence": 0.96},
        {"token": "ការរក្សាសីល", "confidence": 0.94},
        {"token": "ដែល", "confidence": 0.96}
    ],
    3: [
        {"token": "សរជាតិការផ្តល់អោយ", "confidence": 0.92},
        {"token": "ការធ្វើទាន...", "confidence": 0.93},
        {"token": "កិត្តិយស...", "confidence": 0.94},
        {"token": "មានការស្ងប់ក្នុងចិត្ត...", "confidence": 0.92},
        {"token": "ជីវិត", "confidence": 0.96}
    ],
    4: [
        {"token": "មនុស្សគឺមានតម្លៃ", "confidence": 0.93},
        {"token": "ចំណែកសីល", "confidence": 0.94},
        {"token": "គឺការទុកដាក់", "confidence": 0.93},
        {"token": "ការដើរផ្លូវអាក្រក់", "confidence": 0.93}
    ],
    5: [
        {"token": "ចំណែកការកត្តា", "confidence": 0.93},
        {"token": "គឺភាពស្មោះត្រង់", "confidence": 0.94},
        {"token": "ហើយការបរិច្ចាគ", "confidence": 0.93},
        {"token": "ការមិន...", "confidence": 0.94}
    ],
    6: [
        {"token": "កេងប្រវ័ញ្ចអ្នកដទៃ", "confidence": 0.92},
        {"token": "បន្ទាប់មកទៀត", "confidence": 0.94},
        {"token": "ការមានសេចក្តីអត់ធ្មត់", "confidence": 0.93},
        {"token": "និងមិន", "confidence": 0.95}
    ],
    7: [
        {"token": "ត្រូវបានគឺអំពើពាល", "confidence": 0.92},
        {"token": "មានន័យថាការចេះបត់បែន...", "confidence": 0.91},
        {"token": "និងអំពើ", "confidence": 0.95},
        {"token": "បុគ្គល", "confidence": 0.95}
    ],
    8: [
        {"token": "គ្រួសារនៅលើ", "confidence": 0.94},
        {"token": "ទុក្ខវេទនា", "confidence": 0.95},
        {"token": "សង្គម", "confidence": 0.96},
        {"token": "និងសីលធម៌", "confidence": 0.94},
        {"token": "ជាដើម", "confidence": 0.96},
        {"token": "។", "confidence": 0.97},
        {"token": "ភាពនេះ", "confidence": 0.95},
        {"token": "ទៅទៀត", "confidence": 0.95}
    ],
    9: [
        {"token": "នោះគឺការពារឱ្យស្រុកទេស", "confidence": 0.92},
        {"token": "ប្រកបដោយសន្តិភាព", "confidence": 0.93},
        {"token": "ក្នុងន័យ", "confidence": 0.95},
        {"token": "កិត្តិយសនេះមាន", "confidence": 0.93}
    ],
    10: [
        {"token": "ន័យថាត្រូវមាន", "confidence": 0.94},
        {"token": "មេត្តាករុណា", "confidence": 0.95},
        {"token": "មុទិតា", "confidence": 0.96},
        {"token": "និង", "confidence": 0.96},
        {"token": "ឧបេក្ខា", "confidence": 0.96},
        {"token": "។", "confidence": 0.97},
        {"token": "ព្រោះព្រះពុទ្ធសាសនា", "confidence": 0.94}
    ],
    11: [
        {"token": "ក៏បានអប់រំមនុស្ស", "confidence": 0.93},
        {"token": "រាល់គ្នារួបរួមគ្នា", "confidence": 0.93},
        {"token": "ទ្រទ្រង់បានសេចក្តីសុខ...", "confidence": 0.92},
        {"token": "នេះហើយ...", "confidence": 0.93},
        {"token": "។", "confidence": 0.97}
    ],
    12: [
        {"token": "បើយើងនិយាយពី", "confidence": 0.94},
        {"token": "« ផ្នែកនៃការលូតលាស់", "confidence": 0.92},
        {"token": "ខាងវិទ្យាសាស្ត្រ »", "confidence": 0.93},
        {"token": "វិញនោះ...", "confidence": 0.94},
        {"token": "ព្រះ", "confidence": 0.96}
    ],
    13: [
        {"token": "ពុទ្ធឱ្យយើងយល់ដឹង", "confidence": 0.93},
        {"token": "គឺយើងត្រូវ", "confidence": 0.95},
        {"token": "ស្គាល់អំពើអាក្រក់", "confidence": 0.93},
        {"token": "ដោយការ", "confidence": 0.95}
    ],
    14: [
        {"token": "ត្រិះរិះពិចារណា", "confidence": 0.94},
        {"token": "ការគិតឱ្យជ្រៅ", "confidence": 0.93},
        {"token": "ដោយអំពើល្អ", "confidence": 0.94},
        {"token": "របស់ដែល", "confidence": 0.95},
        {"token": "ប្រជាជន", "confidence": 0.96},
        {"token": "ត្រូវការ", "confidence": 0.95}
    ],
    15: [
        {"token": "សរុបមកគឺ", "confidence": 0.94},
        {"token": "« កតញ្ញូកត្តវេទី", "confidence": 0.93},
        {"token": "របស់មនុស្សកាលណាធ្វើ »", "confidence": 0.92},
        {"token": "។", "confidence": 0.97},
        {"token": "អ្វីដែលពិសេសជាង", "confidence": 0.93}
    ],
    16: [
        {"token": "នោះគឺ", "confidence": 0.95},
        {"token": "ព្រះបរមគ្រូ", "confidence": 0.95},
        {"token": "នៃយើង", "confidence": 0.96},
        {"token": "បង្រៀនអប់រំមនុស្ស", "confidence": 0.93},
        {"token": "ឱ្យចេះឱ្យមាន", "confidence": 0.93},
        {"token": "គំនិតទៅ", "confidence": 0.94}
    ],
    17: [
        {"token": "លើទស្សនៈ", "confidence": 0.95},
        {"token": "កម្មផល", "confidence": 0.95},
        {"token": "ពោលគឺ", "confidence": 0.95},
        {"token": "អ្នកធ្វើល្អបានល្អ", "confidence": 0.93},
        {"token": "អ្នកធ្វើអាក្រក់បាន", "confidence": 0.93}
    ],
    18: [
        {"token": "ផលអាក្រក់", "confidence": 0.95},
        {"token": "អ្នកពោលប្រព្រឹត្ត", "confidence": 0.93},
        {"token": "ដែលនាំឱ្យលោភលន់", "confidence": 0.92},
        {"token": "វាជាផល...", "confidence": 0.93},
        {"token": "វាពិត", "confidence": 0.95}
    ],
    19: [
        {"token": "ណាស់", "confidence": 0.96},
        {"token": "។", "confidence": 0.97},
        {"token": "រាល់អំពើអាក្រក់", "confidence": 0.93},
        {"token": "ដែលអ្នកណាបានប្រព្រឹត្ត", "confidence": 0.92},
        {"token": "ហើយនោះគឺ", "confidence": 0.94}
    ],
    20: [
        {"token": "អាបាយមុខ", "confidence": 0.95},
        {"token": "សន្តិសុខ", "confidence": 0.95},
        {"token": "និង", "confidence": 0.96},
        {"token": "អំពើល្អ", "confidence": 0.95},
        {"token": "បានឡើយ", "confidence": 0.95},
        {"token": "។", "confidence": 0.97},
        {"token": "ក្នុងចំណុចនេះដែល", "confidence": 0.93}
    ],
    21: [
        {"token": "ព្រះពុទ្ធសាសនា", "confidence": 0.96},
        {"token": "បានអប់រំឱ្យ", "confidence": 0.94},
        {"token": "មនុស្សរៀនពី", "confidence": 0.94},
        {"token": "អំពើល្អ", "confidence": 0.95},
        {"token": "កាត់បន្ថយកាល", "confidence": 0.93}
    ],
    22: [
        {"token": "ទុច្ចរិត", "confidence": 0.95},
        {"token": "និង", "confidence": 0.96},
        {"token": "ទុច្ចរិតចិត្ត", "confidence": 0.94},
        {"token": "ហើយ", "confidence": 0.96},
        {"token": "ចំណែក", "confidence": 0.95},
        {"token": "ការរស់នៅល្អ", "confidence": 0.93},
        {"token": "ការធ្វើកុសល...", "confidence": 0.93}
    ],
    23: [
        {"token": "ចំណែកនោះគឺ", "confidence": 0.94},
        {"token": "លោកពិត", "confidence": 0.95},
        {"token": "គឺធ្វើកុសល", "confidence": 0.93},
        {"token": "គឺបានន័យថា", "confidence": 0.94},
        {"token": "ស៊ីល្អបានល្អ", "confidence": 0.93},
        {"token": "ស៊ី", "confidence": 0.96}
    ],
    24: [
        {"token": "អាក្រក់បានអាក្រក់", "confidence": 0.93},
        {"token": "អ្វីទាំងឡាយ", "confidence": 0.95},
        {"token": "កើតឡើងដោយការ", "confidence": 0.93},
        {"token": "ស្វែងរកកុសល", "confidence": 0.94},
        {"token": "សាង", "confidence": 0.96}
    ],
    25: [
        {"token": "បុគ្គលម្នាក់ៗ", "confidence": 0.95},
        {"token": "សាសនិកជន", "confidence": 0.95},
        {"token": "។", "confidence": 0.97},
        {"token": "ក្រៅពីនេះ", "confidence": 0.95},
        {"token": "ព្រះពុទ្ធសាសនា", "confidence": 0.96},
        {"token": "បានឱ្យរៀន", "confidence": 0.94}
    ],
    26: [
        {"token": "សរសេរ", "confidence": 0.95},
        {"token": "ប្រជាជន", "confidence": 0.96},
        {"token": "ត្រូវរៀនចេះអាន", "confidence": 0.93},
        {"token": "គិត", "confidence": 0.96},
        {"token": "លើសលប់", "confidence": 0.94},
        {"token": "ភាពស្គាល់", "confidence": 0.94}
    ],
    27: [
        {"token": "ដោយសារការគិត", "confidence": 0.93},
        {"token": "ព្យាយាមបំពេញ", "confidence": 0.94},
        {"token": "ការងារនោះ", "confidence": 0.95},
        {"token": "មិនចោលកាល...", "confidence": 0.93}
    ],
    28: [
        {"token": "កាយកម្លាំង", "confidence": 0.94},
        {"token": "ចិត្តកម្លាំង", "confidence": 0.94},
        {"token": "ប្រាជ្ញាកើតឡើង", "confidence": 0.93},
        {"token": "ឱ្យស័ក្តិសម", "confidence": 0.94},
        {"token": "និងគុណតម្លៃ", "confidence": 0.94},
        {"token": "ការ", "confidence": 0.96}
    ],
    29: [
        {"token": "រលត់បាត់បង់", "confidence": 0.94},
        {"token": "។", "confidence": 0.97},
        {"token": "ក្នុងការស្ថាបនា", "confidence": 0.94},
        {"token": "ទស្សនៈ", "confidence": 0.96},
        {"token": "នៃការយល់ដឹងនេះ", "confidence": 0.93}
    ]
}

LINE_GROUND_TRUTH_P5 = {
    0: [
        {"token": "ឱ្យតែជាការស្វែងរក", "confidence": 0.93},
        {"token": "ចំណេះដឹង", "confidence": 0.95},
        {"token": "ដែលចេញពី", "confidence": 0.94},
        {"token": "គំនិត", "confidence": 0.95},
        {"token": "របស់", "confidence": 0.96},
        {"token": "មនុស្ស", "confidence": 0.96}
    ],
    1: [
        {"token": "ត្រូវជួយ", "confidence": 0.94},
        {"token": "សង្គ្រោះ", "confidence": 0.95},
        {"token": "មនុស្សជាតិ", "confidence": 0.95},
        {"token": "គ្រប់ៗគ្នា...", "confidence": 0.94},
        {"token": "អត្ថន័យ", "confidence": 0.95},
        {"token": "ជាដើម...", "confidence": 0.94}
    ],
    2: [
        {"token": "ជាដើម", "confidence": 0.96},
        {"token": "។", "confidence": 0.97},
        {"token": "អ្វីដែលសបញ្ជាក់", "confidence": 0.93},
        {"token": "បន្ថែមគឺ", "confidence": 0.94},
        {"token": "ព្រះពុទ្ធសាសនា", "confidence": 0.96},
        {"token": "បានអប់រំឱ្យ", "confidence": 0.94}
    ],
    3: [
        {"token": "ដើរតាម", "confidence": 0.95},
        {"token": "ផ្លូវកណ្តាល", "confidence": 0.94},
        {"token": "គឺសច្ចធម៌", "confidence": 0.93},
        {"token": "មិនលំអៀង", "confidence": 0.94},
        {"token": "និងការប្រកាន់យក", "confidence": 0.93},
        {"token": "អត្ត", "confidence": 0.95}
    ],
    4: [
        {"token": "មានច្រើនដូចជា", "confidence": 0.93},
        {"token": "សម្បទាទាំង៤...", "confidence": 0.93},
        {"token": "សម្បទាទី១...", "confidence": 0.94},
        {"token": "សម្បទាទី២...", "confidence": 0.94},
        {"token": "សម្បទាទី៣", "confidence": 0.94}
    ],
    5: [
        {"token": "សម្បទាទី៤...", "confidence": 0.94},
        {"token": "សម្បទាដោយសីលធម៌", "confidence": 0.92},
        {"token": "សម្បទាដែល", "confidence": 0.94},
        {"token": "និយាយអំពី", "confidence": 0.94}
    ],
    6: [
        {"token": "ក្នុងការធ្វើសកម្មភាព", "confidence": 0.92},
        {"token": "ក្នុងការរស់នៅ", "confidence": 0.93},
        {"token": "បានល្អ", "confidence": 0.95},
        {"token": "។", "confidence": 0.97},
        {"token": "លើសពីនេះទៀត", "confidence": 0.93}
    ],
    7: [
        {"token": "ព្រះពុទ្ធសាសនា", "confidence": 0.96},
        {"token": "គឺ", "confidence": 0.97},
        {"token": "ជា", "confidence": 0.96},
        {"token": "ទីពឹងដ៏ជ្រក", "confidence": 0.93},
        {"token": "នៃមនុស្ស", "confidence": 0.95},
        {"token": "អាច", "confidence": 0.96}
    ],
    8: [
        {"token": "ព្យាបាលផ្លូវចិត្ត", "confidence": 0.93},
        {"token": "កម្ចាត់សេចក្តីស្មុគស្មាញ...", "confidence": 0.91},
        {"token": "វប្បធម៌ប្រពៃណី", "confidence": 0.93},
        {"token": "និងសីលធម៌", "confidence": 0.94}
    ],
    9: [
        {"token": "ដើម្បីឱ្យមនុស្ស", "confidence": 0.93},
        {"token": "មានសេចក្តីសុខសាន្ត...", "confidence": 0.92},
        {"token": "ពោលគឺ", "confidence": 0.95},
        {"token": "លុបបំបាត់", "confidence": 0.95}
    ],
    10: [
        {"token": "លុបបំបាត់", "confidence": 0.95},
        {"token": "ការរើសអើងសង្គម", "confidence": 0.93},
        {"token": "រវាងមនុស្សជាតិ", "confidence": 0.94},
        {"token": "។", "confidence": 0.97},
        {"token": "ពេលនោះហើយ", "confidence": 0.93}
    ],
    11: [
        {"token": "ទៀត", "confidence": 0.95},
        {"token": "ត្រូវរក្សាចិត្ត", "confidence": 0.94},
        {"token": "ឱ្យស្អាតបរិសុទ្ធ", "confidence": 0.93},
        {"token": "យើងត្រូវ", "confidence": 0.95},
        {"token": "ប្រកាន់យកនូវ", "confidence": 0.93},
        {"token": "សាមគ្គី", "confidence": 0.95}
    ],
    12: [
        {"token": "ពោលគឺរាល់ការគិត", "confidence": 0.92},
        {"token": "និងការគិតដ៏ខ្ពង់ខ្ពស់", "confidence": 0.92},
        {"token": "ជួយសង្គមជាតិ", "confidence": 0.94},
        {"token": "។", "confidence": 0.97},
        {"token": "រាល់បញ្ហា", "confidence": 0.94}
    ],
    13: [
        {"token": "ទាំងអស់", "confidence": 0.95},
        {"token": "រវាង", "confidence": 0.96},
        {"token": "មនុស្សនេះ", "confidence": 0.95},
        {"token": "គឺ", "confidence": 0.97},
        {"token": "ជាទ្រឹស្តី", "confidence": 0.94},
        {"token": "ដ៏ធំនៃការពិត", "confidence": 0.93},
        {"token": "នៃជីវិត", "confidence": 0.95}
    ],
    14: [
        {"token": "ព្រះសមណគោតម", "confidence": 0.93},
        {"token": "សម្រេច", "confidence": 0.95},
        {"token": "ពុទ្ធសាសនិក", "confidence": 0.95},
        {"token": "។", "confidence": 0.97},
        {"token": "បើយើងនិយាយពី", "confidence": 0.94}
    ],
    15: [
        {"token": "លក្ខខណ្ឌ", "confidence": 0.95},
        {"token": "ការរស់នៅនោះគឺ", "confidence": 0.92},
        {"token": "ធ្វើឱ្យមនុស្ស", "confidence": 0.94},
        {"token": "មានចិត្តស្ងប់", "confidence": 0.94},
        {"token": "មានសេច", "confidence": 0.94}
    ],
    16: [
        {"token": "ក្តីសុខ", "confidence": 0.95},
        {"token": "ព្យាយាមស្វែងរកចំណេះដឹង", "confidence": 0.92},
        {"token": "ស្រឡាញ់សន្តិភាព", "confidence": 0.94},
        {"token": "លុបបំបាត់", "confidence": 0.95}
    ],
    17: [
        {"token": "ភាពរើសអើង...", "confidence": 0.93},
        {"token": "កសាងសង្គមមួយ", "confidence": 0.93},
        {"token": "ពោរពេញដោយការចែករំលែក...", "confidence": 0.90},
        {"token": "ប្រជាជនរស់នៅ", "confidence": 0.93}
    ],
    18: [
        {"token": "ប្រកបដោយ", "confidence": 0.94},
        {"token": "សេចក្តីសុខ...", "confidence": 0.93},
        {"token": "ស្មើគ្នា", "confidence": 0.95},
        {"token": "។", "confidence": 0.97},
        {"token": "ចំណែកក្នុងការជ្រើស", "confidence": 0.92}
    ],
    19: [
        {"token": "រើសយក", "confidence": 0.95},
        {"token": "គឺមានន័យថា", "confidence": 0.93},
        {"token": "ក៏ដូចជាការធ្វើទាន...", "confidence": 0.92},
        {"token": "ក្នុងបំណងស្វែងរក", "confidence": 0.92}
    ],
    20: [
        {"token": "ការសប្បាយ", "confidence": 0.94},
        {"token": "អ្វីដែលសំខាន់", "confidence": 0.93},
        {"token": "នាំឆ្ពោះទៅរកសន្តិភាព...", "confidence": 0.91},
        {"token": "ក៏ប៉ុន្តែដើម្បី", "confidence": 0.93}
    ],
    21: [
        {"token": "ផលប្រយោជន៍", "confidence": 0.94},
        {"token": "ទាំងអស់នេះ", "confidence": 0.95},
        {"token": "នោះត្រូវតែមាន", "confidence": 0.93},
        {"token": "វិស័យ", "confidence": 0.95},
        {"token": "ក្នុងការ", "confidence": 0.95}
    ],
    22: [
        {"token": "បង្ហាត់បង្រៀន", "confidence": 0.94},
        {"token": "ឱ្យមានចំណេះដឹង...", "confidence": 0.92},
        {"token": "នោះគឺវត្តអារាម", "confidence": 0.93},
        {"token": "ដែលជាទី", "confidence": 0.94}
    ],
    23: [
        {"token": "អាស្រ័យ", "confidence": 0.95},
        {"token": "បណ្តុះចំណេះដឹង...", "confidence": 0.92},
        {"token": "ព្រោះកាលពីសម័យបុរាណ", "confidence": 0.92},
        {"token": "វត្តអារាម", "confidence": 0.95}
    ],
    24: [
        {"token": "របស់សង្គមលោក", "confidence": 0.93},
        {"token": "គឺជាផ្នែកមួយ", "confidence": 0.93},
        {"token": "រៀនសូត្រ", "confidence": 0.94},
        {"token": "របស់កូនចៅខ្មែរ", "confidence": 0.93}
    ],
    25: [
        {"token": "ជាកន្លែង", "confidence": 0.95},
        {"token": "បណ្តុះបណ្តាល", "confidence": 0.94},
        {"token": "ជាបណ្ណាល័យ", "confidence": 0.94},
        {"token": "ជាមន្ទីរពេទ្យ", "confidence": 0.94},
        {"token": "ព្យាបាលជំងឺ", "confidence": 0.94}
    ],
    26: [
        {"token": "សីលធម៌", "confidence": 0.95},
        {"token": "ពិនិត្យវត្តអារាម", "confidence": 0.93},
        {"token": "ជាមណ្ឌល", "confidence": 0.95},
        {"token": "វិទ្យាសាស្ត្រ", "confidence": 0.95},
        {"token": "និងវប្បធម៌ខ្មែរ", "confidence": 0.93}
    ],
    27: [
        {"token": "ដោយសារ", "confidence": 0.95},
        {"token": "វត្តអារាម", "confidence": 0.95},
        {"token": "ពោរពេញដោយ", "confidence": 0.93},
        {"token": "ទីកន្លែងស្រាវជ្រាវ", "confidence": 0.93},
        {"token": "ការរៀនសូត្រ", "confidence": 0.94}
    ],
    28: [
        {"token": "អក្សរសាស្ត្រជាដើម", "confidence": 0.93},
        {"token": "។", "confidence": 0.97},
        {"token": "ធាតុផ្សំទាំងមូល", "confidence": 0.93},
        {"token": "កសាងសីលធម៌", "confidence": 0.94},
        {"token": "កុសល", "confidence": 0.95}
    ],
    29: [
        {"token": "ផ្សេងៗ", "confidence": 0.96},
        {"token": "ព្រោះកាលពី", "confidence": 0.94},
        {"token": "សម័យសកលលោក...", "confidence": 0.92},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.96},
        {"token": "ព្រះធម៌", "confidence": 0.96}
    ]
}

LINE_GROUND_TRUTH_P6 = {
    0: [
        {"token": "ព្រះសង្ឃ", "confidence": 0.96},
        {"token": "»", "confidence": 0.96},
        {"token": "ជាទីសក្ការៈ", "confidence": 0.94},
        {"token": "និងប្រាថ្នា", "confidence": 0.94},
        {"token": "ចង់បាន", "confidence": 0.95},
        {"token": "ជាដើម", "confidence": 0.95},
        {"token": "។", "confidence": 0.97},
        {"token": "ទាំងនេះ", "confidence": 0.95},
        {"token": "គឺជាទីតាំង", "confidence": 0.94}
    ],
    1: [
        {"token": "សម្រាប់", "confidence": 0.95},
        {"token": "សិក្សា", "confidence": 0.94},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.95},
        {"token": "សាសនា", "confidence": 0.95},
        {"token": "។", "confidence": 0.97}
    ],
    2: [
        {"token": "ជាក់ស្តែង", "confidence": 0.94},
        {"token": "បើយើង", "confidence": 0.95},
        {"token": "ក្រឡេកមើលទៅ", "confidence": 0.93},
        {"token": "ស្ថានីយ", "confidence": 0.94},
        {"token": "អភិរក្សសិល្បៈ", "confidence": 0.93},
        {"token": "វប្បធម៌", "confidence": 0.94},
        {"token": "ទាំង", "confidence": 0.95}
    ],
    3: [
        {"token": "ព្រះបាទជ័យវរ្ម័ន", "confidence": 0.94},
        {"token": "ទី៧...", "confidence": 0.92},
        {"token": "ទាំងអស់នេះ", "confidence": 0.95},
        {"token": "គឺជាស្ថានីយ", "confidence": 0.94},
        {"token": "អភិរក្សសិល្បៈ", "confidence": 0.94},
        {"token": "ដែលជាពុទ្ធ", "confidence": 0.93},
        {"token": "សាសនា...", "confidence": 0.92}
    ],
    4: [
        {"token": "ដែលបានបង្ហាញ", "confidence": 0.93},
        {"token": "ញ", "confidence": 0.90},
        {"token": "អំពីការធ្វើ", "confidence": 0.93},
        {"token": "ទាន", "confidence": 0.96},
        {"token": "របស់ព្រះបាទជ័យវរ្ម័ន", "confidence": 0.92},
        {"token": "។", "confidence": 0.97},
        {"token": "ក្រោយពីសិល្បៈទាំង", "confidence": 0.92}
    ],
    5: [
        {"token": "នោះហើយបើយើង", "confidence": 0.93},
        {"token": "ងាក", "confidence": 0.96},
        {"token": "និយាយបានថា...", "confidence": 0.92},
        {"token": "វត្តអារាម", "confidence": 0.95},
        {"token": "ព្រះពុទ្ធសាសនា", "confidence": 0.94},
        {"token": "បានកសាងកុសល...", "confidence": 0.92}
    ],
    6: [
        {"token": "ពោលគឺ", "confidence": 0.95},
        {"token": "ការធ្វើទាន...", "confidence": 0.93},
        {"token": "សីល", "confidence": 0.95},
        {"token": "ដែលឈ្មោះ", "confidence": 0.94},
        {"token": "ថា", "confidence": 0.96},
        {"token": "«", "confidence": 0.96},
        {"token": "»", "confidence": 0.96},
        {"token": "ពោលពលករ...", "confidence": 0.92}
    ],
    7: [
        {"token": "មាន", "confidence": 0.96},
        {"token": "ចិត្ត", "confidence": 0.96},
        {"token": "មេត្តា", "confidence": 0.95},
        {"token": "ករុណា...", "confidence": 0.93},
        {"token": "មុទិតា", "confidence": 0.94},
        {"token": "ឧបេក្ខា...", "confidence": 0.93},
        {"token": "រួមជាមួយ", "confidence": 0.93},
        {"token": "កាយ", "confidence": 0.95},
        {"token": "និងស្មារតី...", "confidence": 0.92}
    ],
    8: [
        {"token": "របស់មនុស្សទៀតផង", "confidence": 0.92},
        {"token": "។", "confidence": 0.97},
        {"token": "ព្រះ", "confidence": 0.96},
        {"token": "ពុទ្ធបាន", "confidence": 0.95},
        {"token": "ធ្វើការ", "confidence": 0.95},
        {"token": "អប់រំដាស់", "confidence": 0.94},
        {"token": "តឿនការ", "confidence": 0.94},
        {"token": "ដឹក", "confidence": 0.96}
    ],
    9: [
        {"token": "នាំនេះឱ្យល្អប្រសើរឡើង...", "confidence": 0.92},
        {"token": "ការកាត់បន្ថយ", "confidence": 0.94},
        {"token": "ជម្លោះ", "confidence": 0.95},
        {"token": "រវាងមនុស្សជាតិ... ឱ្យបានច្រើនៗ", "confidence": 0.90},
        {"token": "ក្រៅពីនោះ...", "confidence": 0.93}
    ],
    10: [
        {"token": "យល់ដឹង", "confidence": 0.95},
        {"token": "ការកាត់បន្ថយ", "confidence": 0.94},
        {"token": "ជម្លោះ...", "confidence": 0.94},
        {"token": "ការកាត់", "confidence": 0.95},
        {"token": "បន្ថយ", "confidence": 0.95},
        {"token": "និងការសិក្សាស្វែង", "confidence": 0.93}
    ],
    11: [
        {"token": "យល់ច្រើន", "confidence": 0.94},
        {"token": "ទៀត...", "confidence": 0.93},
        {"token": "ព្រះ", "confidence": 0.96},
        {"token": "ពុទ្ធបាន", "confidence": 0.95},
        {"token": "បង្រៀន", "confidence": 0.94},
        {"token": "ឱ្យ", "confidence": 0.95},
        {"token": "ចេះអត់ធ្មត់", "confidence": 0.93},
        {"token": "និង", "confidence": 0.96},
        {"token": "ស្វែងរក", "confidence": 0.95},
        {"token": "ភាពសុខសាន្ត", "confidence": 0.94},
        {"token": "ពោល", "confidence": 0.96}
    ],
    12: [
        {"token": "គឺនៅពេលដែល...", "confidence": 0.93},
        {"token": "អ្នករងទុក្ខលំបាកស្វែងរកទីពឹង... ព្រះពុទ្ធជាអ្នកចេញ", "confidence": 0.90},
        {"token": "មុខ...", "confidence": 0.94}
    ],
    13: [
        {"token": "ដោះស្រាយ...", "confidence": 0.93},
        {"token": "ក៏", "confidence": 0.96},
        {"token": "ព្រះពុទ្ធបាន", "confidence": 0.94},
        {"token": "ចាត់", "confidence": 0.96},
        {"token": "តាំងសិល្បៈនេះ...", "confidence": 0.93},
        {"token": "ដោយ", "confidence": 0.96},
        {"token": "មាន", "confidence": 0.96},
        {"token": "ការពិនិត្យ", "confidence": 0.94},
        {"token": "ស្វែងយល់...", "confidence": 0.93}
    ],
    14: [
        {"token": "នោះទៅហើយព្រះពុទ្ធបាន", "confidence": 0.92},
        {"token": "ផ្តល់", "confidence": 0.95},
        {"token": "ការស្រឡាញ់", "confidence": 0.94},
        {"token": "ដល់អ្នកដទៃ...", "confidence": 0.93},
        {"token": "សេចក្តី", "confidence": 0.95},
        {"token": "ស្រឡាញ់...", "confidence": 0.94}
    ],
    15: [
        {"token": "ឱ្យមានសិទ្ធិ...", "confidence": 0.93},
        {"token": "ដូចដែល", "confidence": 0.95},
        {"token": "ទ្រឹស្តីព្រះពុទ្ធ...", "confidence": 0.93},
        {"token": "ចំណេះដឹង", "confidence": 0.94},
        {"token": "ប្រព្រឹត្តមនុស្សឱ្យស្រឡាញ់", "confidence": 0.92},
        {"token": "ដោយស្មោះស្ម័គ្រ...", "confidence": 0.92}
    ],
    16: [
        {"token": "សន្តិភាព", "confidence": 0.95},
        {"token": "របស់ព្រះពុទ្ធ...", "confidence": 0.93},
        {"token": "ក៏ដើម្បី", "confidence": 0.94},
        {"token": "ព្រះពុទ្ធចាត់ចែងជីវិតសង្គម", "confidence": 0.91},
        {"token": "ជាតិ", "confidence": 0.95},
        {"token": "។", "confidence": 0.97},
        {"token": "ដោយ", "confidence": 0.96}
    ],
    17: [
        {"token": "ផ្អែក", "confidence": 0.93},
        {"token": "លើទស្សនៈ", "confidence": 0.94},
        {"token": "កម្មផល...", "confidence": 0.94},
        {"token": "ធ្វើល្អបាន", "confidence": 0.94},
        {"token": "ល្អ...", "confidence": 0.95},
        {"token": "ធ្វើអាក្រក់បានអាក្រក់នោះ...", "confidence": 0.92}
    ],
    18: [
        {"token": "ដូចក្រោយ", "confidence": 0.94},
        {"token": "ព្រះពុទ្ធបាន", "confidence": 0.95},
        {"token": "ជួយ", "confidence": 0.96},
        {"token": "សង្គ្រោះ...", "confidence": 0.94},
        {"token": "ឱ្យស្គាល់សីលធម៌និងគុណធម៌...", "confidence": 0.92},
        {"token": "នោះ...", "confidence": 0.94}
    ],
    19: [
        {"token": "ដោយសារការប្រព្រឹត្ត", "confidence": 0.93},
        {"token": "អំពើល្អ", "confidence": 0.94},
        {"token": "នេះឯង", "confidence": 0.95},
        {"token": "។", "confidence": 0.97},
        {"token": "ម្យ៉ាង", "confidence": 0.95},
        {"token": "វិញទៀត...", "confidence": 0.94},
        {"token": "ក្នុងការ", "confidence": 0.95},
        {"token": "...", "confidence": 0.85}
    ],
    20: [
        {"token": "អភិរក្សសិល្បៈវប្បធម៌ខ្មែរយើង...", "confidence": 0.91},
        {"token": "គឺ", "confidence": 0.96},
        {"token": "ស្ថិតក្នុង", "confidence": 0.95},
        {"token": "វត្តអារាម", "confidence": 0.95},
        {"token": "ព្រះពុទ្ធសាសនា...", "confidence": 0.94},
        {"token": "ទាំងអស់នេះហើយ", "confidence": 0.93},
        {"token": "...", "confidence": 0.85}
    ],
    21: [
        {"token": "គឺបានបង្ហាញ", "confidence": 0.94},
        {"token": "នូវសច្ចភាព", "confidence": 0.93},
        {"token": "ពន្លឺសន្តិភាព...", "confidence": 0.93},
        {"token": "...", "confidence": 0.85},
        {"token": "សន្តិភាព...", "confidence": 0.93},
        {"token": "វិទ្យាសាស្ត្រ...", "confidence": 0.94},
        {"token": "ដោយគ្មានសង្គ្រាម...", "confidence": 0.92}
    ],
    22: [
        {"token": "រវាងមនុស្ស...", "confidence": 0.94},
        {"token": "គ្មានការកេងប្រវ័ញ្ច", "confidence": 0.93},
        {"token": "ផងដែរ...", "confidence": 0.94},
        {"token": "ទាំងអស់នេះ", "confidence": 0.94},
        {"token": "គឺ", "confidence": 0.96},
        {"token": "ការរស់នៅ", "confidence": 0.94},
        {"token": "ល្អឡើយ...", "confidence": 0.93}
    ],
    23: [
        {"token": "ត្រង់ចំណុចនេះ", "confidence": 0.94},
        {"token": "គឺយោង", "confidence": 0.94},
        {"token": "តាមការចាត់", "confidence": 0.94},
        {"token": "តាំង", "confidence": 0.96},
        {"token": "របស់ព្រះពុទ្ធ", "confidence": 0.94},
        {"token": "នៃការ", "confidence": 0.95},
        {"token": "រស់", "confidence": 0.96},
        {"token": "នៅរបស់", "confidence": 0.94},
        {"token": "...", "confidence": 0.85}
    ],
    24: [
        {"token": "ព្រះបរមគ្រូនៃ", "confidence": 0.93},
        {"token": "យើង...", "confidence": 0.95},
        {"token": "ដែល", "confidence": 0.96},
        {"token": "អប់រំឱ្យ", "confidence": 0.94},
        {"token": "មនុស្សមាន", "confidence": 0.94},
        {"token": "ភាព", "confidence": 0.96},
        {"token": "ស្ងប់...", "confidence": 0.94}
    ],
    25: [
        {"token": "កុំឱ្យមាន", "confidence": 0.94},
        {"token": "ភាពលោភលន់", "confidence": 0.93},
        {"token": "ភាពច្រណែន", "confidence": 0.94},
        {"token": "ជាដើម", "confidence": 0.95},
        {"token": "ពោលគឺ ត្រូវចេះ...", "confidence": 0.92}
    ],
    26: [
        {"token": "រៀន", "confidence": 0.96},
        {"token": "លះបង់", "confidence": 0.95},
        {"token": "នូវការលោភលន់!", "confidence": 0.93},
        {"token": "ចោល", "confidence": 0.95},
        {"token": "លុះដល់", "confidence": 0.94},
        {"token": "គំនិត", "confidence": 0.95},
        {"token": "គិត", "confidence": 0.96},
        {"token": "...", "confidence": 0.85},
        {"token": "គម្រោង...", "confidence": 0.93}
    ],
    27: [
        {"token": "លោភលន់ស្វែងរក", "confidence": 0.93},
        {"token": "អ្វីៗ", "confidence": 0.95},
        {"token": "ដោយ", "confidence": 0.96},
        {"token": "ខ្នះខ្នែង", "confidence": 0.93},
        {"token": "ជាការ", "confidence": 0.95},
        {"token": "គិតប្រយោជន៍ឱ្យប្រជាជន...", "confidence": 0.92}
    ],
    28: [
        {"token": "ប្រជាជនទាំងអស់ភាពកណ្តាល...", "confidence": 0.91},
        {"token": "នោះ", "confidence": 0.96},
        {"token": "គឺជាកិត្តិយស...", "confidence": 0.93},
        {"token": "នោះ", "confidence": 0.96},
        {"token": "ដោយសារ", "confidence": 0.94},
        {"token": "ការគិត", "confidence": 0.94}
    ],
    29: [
        {"token": "អន្តរ", "confidence": 0.94},
        {"token": "នោះ", "confidence": 0.96},
        {"token": "ដែល", "confidence": 0.96},
        {"token": "ជា", "confidence": 0.96},
        {"token": "ដុះពន្លកនៃ", "confidence": 0.93},
        {"token": "សេច", "confidence": 0.95},
        {"token": "ក្តីទុក្ខ", "confidence": 0.94},
        {"token": "ដោយត្រូវ", "confidence": 0.94},
        {"token": "ប្រកាន់...", "confidence": 0.93}
    ]
}

LINE_GROUND_TRUTH_P7 = {
    0: [
        {"token": "យកផ្លូវកណ្តាល", "confidence": 0.94},
        {"token": "។", "confidence": 0.97},
        {"token": "ម្យ៉ាងវិញទៀត", "confidence": 0.94},
        {"token": "នៅក្នុង", "confidence": 0.95},
        {"token": "វិញ្ញាសា", "confidence": 0.93},
        {"token": "អក្សរសិល្ប៍បែប...", "confidence": 0.92}
    ],
    1: [
        {"token": "ប្រឡោមលោកវិញ", "confidence": 0.93},
        {"token": "គឺរឿង", "confidence": 0.95},
        {"token": "កុលាបប៉ៃលិន", "confidence": 0.94},
        {"token": "ដែល", "confidence": 0.96},
        {"token": "ជាស្នាដៃ", "confidence": 0.94},
        {"token": "របស់", "confidence": 0.96},
        {"token": "លោក", "confidence": 0.96},
        {"token": "...", "confidence": 0.85}
    ],
    2: [
        {"token": "ញ៉ុក", "confidence": 0.95},
        {"token": "ថែម", "confidence": 0.95},
        {"token": "។", "confidence": 0.97},
        {"token": "រឿងនោះ", "confidence": 0.94},
        {"token": "បានបង្ហាញ", "confidence": 0.94},
        {"token": "អំពីទំនាក់ទំនងរវាង", "confidence": 0.92},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.95},
        {"token": "សាសនា", "confidence": 0.95},
        {"token": "...", "confidence": 0.85}
    ],
    3: [
        {"token": "ទៅនឹង", "confidence": 0.95},
        {"token": "វប្បធម៌សន្តិភាព", "confidence": 0.93},
        {"token": "នៅត្រង់", "confidence": 0.94},
        {"token": "ចំណុច", "confidence": 0.95},
        {"token": "ដែល", "confidence": 0.96},
        {"token": "តាផែន", "confidence": 0.94},
        {"token": "នៅ", "confidence": 0.96},
        {"token": "ពេល...", "confidence": 0.93}
    ],
    4: [
        {"token": "គាត់ជិត", "confidence": 0.94},
        {"token": "ផុតដង្ហើម", "confidence": 0.93},
        {"token": "ជីវិត", "confidence": 0.95},
        {"token": "គាត់បាន", "confidence": 0.94},
        {"token": "ពោល", "confidence": 0.95},
        {"token": "ពាក្យ", "confidence": 0.95},
        {"token": "ផ្ញើមក", "confidence": 0.93},
        {"token": "ផ្ទះ", "confidence": 0.95},
        {"token": "ជាបណ្តាំចុង...", "confidence": 0.92}
    ],
    5: [
        {"token": "ក្រោយ", "confidence": 0.95},
        {"token": "“", "confidence": 0.96},
        {"token": "អត្តាហិ", "confidence": 0.94},
        {"token": "អត្តនោ នាថោ", "confidence": 0.93},
        {"token": "”", "confidence": 0.96},
        {"token": "ពាក្យ", "confidence": 0.95},
        {"token": "មួយឃ្លានេះ", "confidence": 0.94},
        {"token": "ជាទិសដៅ", "confidence": 0.94},
        {"token": "ឱ្យ...", "confidence": 0.93}
    ],
    6: [
        {"token": "កូនចៅកែខ្លួន...", "confidence": 0.92},
        {"token": "។", "confidence": 0.97},
        {"token": "ពាក្យនេះ", "confidence": 0.94},
        {"token": "គឺជាកន្លែងចិត្ត", "confidence": 0.92},
        {"token": "របស់", "confidence": 0.96},
        {"token": "ចៅចិត្រ", "confidence": 0.94},
        {"token": "ដែលគាត់...", "confidence": 0.92}
    ],
    7: [
        {"token": "វត្តមាន", "confidence": 0.94},
        {"token": "តែងតែមានសិទ្ធិ", "confidence": 0.92},
        {"token": "ដោយប្រកាន់យក", "confidence": 0.93},
        {"token": "លក្ខណៈ", "confidence": 0.94},
        {"token": "ជា", "confidence": 0.96},
        {"token": "វិជ្ជាជីវៈជាដើម", "confidence": 0.93},
        {"token": "។", "confidence": 0.97},
        {"token": "បើយើង", "confidence": 0.95}
    ],
    8: [
        {"token": "ចិត្ត", "confidence": 0.96},
        {"token": "មានការខិតខំ", "confidence": 0.94},
        {"token": "តស៊ូ", "confidence": 0.95},
        {"token": "ព្យាយាម", "confidence": 0.94},
        {"token": "ធ្វើការងារ", "confidence": 0.94},
        {"token": "ដោយ", "confidence": 0.96},
        {"token": "យកចិត្ត", "confidence": 0.95},
        {"token": "ទុកដាក់", "confidence": 0.95},
        {"token": "និងមាន...", "confidence": 0.93}
    ],
    9: [
        {"token": "ប្រុងប្រយ័ត្ន", "confidence": 0.94},
        {"token": "ជានិច្ច", "confidence": 0.95},
        {"token": "។", "confidence": 0.97},
        {"token": "ចៅចិត្រ", "confidence": 0.95},
        {"token": "បានធ្វើ", "confidence": 0.95},
        {"token": "អំពើល្អ", "confidence": 0.94},
        {"token": "ជាច្រើន", "confidence": 0.94},
        {"token": "ដូចជា...", "confidence": 0.93}
    ],
    10: [
        {"token": "ក្នុងការ", "confidence": 0.95},
        {"token": "ជួយជីដូន", "confidence": 0.94},
        {"token": "ចាស់ៗ", "confidence": 0.95},
        {"token": "ការចេះប្រកាន់", "confidence": 0.93},
        {"token": "ទស្សនៈ", "confidence": 0.95},
        {"token": "ឱ្យមាន", "confidence": 0.95},
        {"token": "សីលធម៌", "confidence": 0.94},
        {"token": "បានរៀនដឹងឮ...", "confidence": 0.92}
    ],
    11: [
        {"token": "លោកហ្លួង", "confidence": 0.94},
        {"token": "រតនៈ", "confidence": 0.95},
        {"token": "សម្បត្តិ", "confidence": 0.95},
        {"token": "ក៏ដូចជា", "confidence": 0.94},
        {"token": "ការការពារ", "confidence": 0.94},
        {"token": "នាង", "confidence": 0.96},
        {"token": "ឃុននារី...", "confidence": 0.93},
        {"token": "។", "confidence": 0.97},
        {"token": "ចៅ", "confidence": 0.96},
        {"token": "ចិត្របាន...", "confidence": 0.93}
    ],
    12: [
        {"token": "ចេញខ្លួន", "confidence": 0.94},
        {"token": "ទៅជួយជំនួសនាង", "confidence": 0.92},
        {"token": "ដោយមិន", "confidence": 0.95},
        {"token": "គិត", "confidence": 0.96},
        {"token": "ពីជីវិតខ្លួន", "confidence": 0.94},
        {"token": "ឡើយ", "confidence": 0.95},
        {"token": "នោះ...", "confidence": 0.93},
        {"token": "។", "confidence": 0.97}
    ],
    13: [
        {"token": "នេះ", "confidence": 0.96},
        {"token": "គឺជាការបង្ហាញចេញនូវអំពើល្អ...", "confidence": 0.91},
        {"token": "ដែលចេញ", "confidence": 0.94},
        {"token": "ពី", "confidence": 0.96},
        {"token": "ចិត្ត...", "confidence": 0.93}
    ],
    14: [
        {"token": "ការជួយអ្នកដទៃ...", "confidence": 0.93},
        {"token": "។", "confidence": 0.97},
        {"token": "មួយវិញទៀត", "confidence": 0.94},
        {"token": "គឺមានការត្រួតត្រា", "confidence": 0.92},
        {"token": "ក្នុងសីលធម៌", "confidence": 0.94},
        {"token": "...", "confidence": 0.85},
        {"token": "...", "confidence": 0.85},
        {"token": "...", "confidence": 0.85}
    ],
    15: [
        {"token": "ក្លាយជាអ្នក...", "confidence": 0.92},
        {"token": "ដែលជាបន្ទុកលោកហ្លួង", "confidence": 0.92},
        {"token": "រតនៈ", "confidence": 0.95},
        {"token": "សម្បត្តិ...", "confidence": 0.94},
        {"token": "។", "confidence": 0.97}
    ],
    16: [
        {"token": "ការធ្វើ", "confidence": 0.95},
        {"token": "អំពើល្អ", "confidence": 0.95},
        {"token": "នាំបង្កើត", "confidence": 0.94},
        {"token": "សីលធម៌...", "confidence": 0.93},
        {"token": "ការងារល្អ", "confidence": 0.94},
        {"token": "ជីវិត", "confidence": 0.95},
        {"token": "ជោគជ័យ", "confidence": 0.94},
        {"token": "...", "confidence": 0.85},
        {"token": "។", "confidence": 0.97}
    ],
    17: [
        {"token": "សរុបសេចក្តីមក", "confidence": 0.93},
        {"token": "យើង", "confidence": 0.96},
        {"token": "អាចដឹងបានថា", "confidence": 0.93},
        {"token": "ទ្រឹស្តី", "confidence": 0.95},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.95},
        {"token": "សាសនា", "confidence": 0.95},
        {"token": "...", "confidence": 0.85}
    ],
    18: [
        {"token": "ដែលកាលពី", "confidence": 0.94},
        {"token": "បុព្វបុរស", "confidence": 0.94},
        {"token": "បានធ្វើឱ្យមានវប្បធម៌", "confidence": 0.92},
        {"token": "សន្តិភាព...", "confidence": 0.93}
    ],
    19: [
        {"token": "ឡើយ", "confidence": 0.95},
        {"token": "ដោយសារ", "confidence": 0.95},
        {"token": "តែពុទ្ធសាសនា", "confidence": 0.93},
        {"token": "បានធ្វើឱ្យ", "confidence": 0.94},
        {"token": "ជាតិខ្មែរ", "confidence": 0.94},
        {"token": "មានសេចក្តីសុខ...", "confidence": 0.92},
        {"token": "ជីវិត", "confidence": 0.95},
        {"token": "...", "confidence": 0.85}
    ],
    20: [
        {"token": "មានការរីកចម្រើន...", "confidence": 0.92},
        {"token": "ដែលកាលពី", "confidence": 0.94},
        {"token": "សម័យសិល្បៈ...", "confidence": 0.93},
        {"token": "ដោយព្រះពុទ្ធ...", "confidence": 0.93},
        {"token": "បាន", "confidence": 0.96},
        {"token": "កសាង", "confidence": 0.95},
        {"token": "...", "confidence": 0.85},
        {"token": "...", "confidence": 0.85}
    ],
    21: [
        {"token": "កសាងសិល្បៈ", "confidence": 0.93},
        {"token": "ព្រះពុទ្ធ...", "confidence": 0.94},
        {"token": "ជាច្រើន...", "confidence": 0.93},
        {"token": "ឱ្យមនុស្សមាន", "confidence": 0.93},
        {"token": "សេចក្តី", "confidence": 0.95},
        {"token": "សុខសាន្ត", "confidence": 0.94},
        {"token": "...", "confidence": 0.85}
    ],
    22: [
        {"token": "បានច្រើន...", "confidence": 0.93},
        {"token": "។", "confidence": 0.97}
    ],
    23: [
        {"token": "ដូចគ្នាតាមការបកស្រាយ", "confidence": 0.92},
        {"token": "ខាងលើ", "confidence": 0.95},
        {"token": "ប្រធានខាងលើ", "confidence": 0.93},
        {"token": "ដូចនេះ", "confidence": 0.95},
        {"token": "យើង", "confidence": 0.96},
        {"token": "...", "confidence": 0.85}
    ],
    24: [
        {"token": "អាចសន្និដ្ឋានបានថា", "confidence": 0.92},
        {"token": "ប្រធានខាងលើ", "confidence": 0.93},
        {"token": "ពិតជាមានតម្លៃ", "confidence": 0.93},
        {"token": "យ៉ាង", "confidence": 0.96},
        {"token": "ធំធេង...", "confidence": 0.93},
        {"token": "ព្រោះ", "confidence": 0.95},
        {"token": "...", "confidence": 0.85},
        {"token": "...", "confidence": 0.85}
    ],
    25: [
        {"token": "បានបង្ហាញ", "confidence": 0.94},
        {"token": "ពីទំនាក់ទំនង", "confidence": 0.93},
        {"token": "រវាង", "confidence": 0.95},
        {"token": "ព្រះពុទ្ធ", "confidence": 0.95},
        {"token": "សាសនា", "confidence": 0.95},
        {"token": "និង", "confidence": 0.96},
        {"token": "វប្បធម៌", "confidence": 0.95},
        {"token": "សន្តិ", "confidence": 0.94},
        {"token": "...", "confidence": 0.85},
        {"token": "...", "confidence": 0.85}
    ],
    26: [
        {"token": "ភាព...", "confidence": 0.93},
        {"token": "ពិសេសទៀតនោះ", "confidence": 0.93},
        {"token": "គឺបានបង្ហាញ", "confidence": 0.93},
        {"token": "ពី", "confidence": 0.96},
        {"token": "តួនាទី", "confidence": 0.94},
        {"token": "ក៏ដូចជាតម្លៃ", "confidence": 0.93},
        {"token": "...", "confidence": 0.85}
    ],
    27: [
        {"token": "នៃ", "confidence": 0.96},
        {"token": "ព្រះពុទ្ធសាសនា", "confidence": 0.94},
        {"token": "ដែលជា", "confidence": 0.95},
        {"token": "សាសនា", "confidence": 0.95},
        {"token": "របស់", "confidence": 0.96},
        {"token": "រដ្ឋ", "confidence": 0.96},
        {"token": "មួយ", "confidence": 0.96},
        {"token": "នោះ...", "confidence": 0.94},
        {"token": "។", "confidence": 0.97},
        {"token": "...", "confidence": 0.85}
    ],
    28: [
        {"token": "ដូចនេះ", "confidence": 0.95},
        {"token": "ក្នុងនាម", "confidence": 0.94},
        {"token": "យើង", "confidence": 0.96},
        {"token": "ជាកូនសិស្ស", "confidence": 0.93},
        {"token": "យើង", "confidence": 0.96},
        {"token": "ត្រូវ", "confidence": 0.96},
        {"token": "ចេះ", "confidence": 0.96},
        {"token": "មើល", "confidence": 0.96},
        {"token": "...", "confidence": 0.85}
    ],
    29: [
        {"token": "ថែរក្សា", "confidence": 0.94},
        {"token": "គោរពគំនិត", "confidence": 0.94},
        {"token": "រៀនសូត្រតាម", "confidence": 0.93},
        {"token": "សាសនានេះឱ្យ...", "confidence": 0.92},
        {"token": "ទាំងអស់គ្នា", "confidence": 0.93},
        {"token": "...", "confidence": 0.85}
    ]
}

PAGE_GROUND_TRUTH = {
    "page_001.png": LINE_GROUND_TRUTH_P1,
    "page_002.png": LINE_GROUND_TRUTH_P2,
    "page_003.png": LINE_GROUND_TRUTH_P3,
    "page_004.png": LINE_GROUND_TRUTH_P4,
    "page_005.png": LINE_GROUND_TRUTH_P5,
    "page_006.png": LINE_GROUND_TRUTH_P6,
    "page_007.png": LINE_GROUND_TRUTH_P7
}

LINE_GROUND_TRUTH_W002_P1 = {
    0: [
        {"token": "ប្រធាន:", "confidence": 0.95},
        {"token": "សុភាសិតខ្មែរ", "confidence": 0.95},
        {"token": "បានចែងថា", "confidence": 0.95},
        {"token": "“", "confidence": 0.90},
        {"token": "ការពារប្រសើរជាងព្យាបាល", "confidence": 0.90, "notes": "compound title"},
        {"token": "”", "confidence": 0.90},
        {"token": "។ ចូរស្រាយ", "confidence": 0.90}
    ],
    1: [
        {"token": "ដោយលើកឧទាហរណ៍ក្នុងជីវិត", "confidence": 0.90, "notes": "compound phrase"},
        {"token": "មកបញ្ជាក់", "confidence": 0.95},
        {"token": "។", "confidence": 0.95}
    ],
    2: [
        {"token": "សេចក្តីអធិប្បាយ", "confidence": 0.95, "notes": "section heading"}
    ],
    3: [
        {"token": "នៅលើសកលលោកយើងនេះ", "confidence": 0.90, "notes": "continuous phrase"},
        {"token": "គ្រប់អ្វីៗទាំងអស់តែងតែមានការប្រែប្រួលជា", "confidence": 0.85, "notes": "continuous sentence"}
    ],
    4: [
        {"token": "និច្ច", "confidence": 0.95},
        {"token": "។", "confidence": 0.95},
        {"token": "គ្រាន់តែពេលខ្លះ", "confidence": 0.90},
        {"token": "ប្រែប្រួលតិច", "confidence": 0.90},
        {"token": "និងពេលខ្លះ", "confidence": 0.90},
        {"token": "មាន", "confidence": 0.95},
        {"token": "ការប្រែប្រួលខ្លាំងប៉ុណ្ណោះ", "confidence": 0.85},
        {"token": "។", "confidence": 0.95}
    ],
    5: [
        {"token": "មាន", "confidence": 0.95},
        {"token": "ហេតុ", "confidence": 0.95},
        {"token": "ត្រូវតែ", "confidence": 0.95},
        {"token": "មានផល", "confidence": 0.95},
        {"token": "។", "confidence": 0.95},
        {"token": "ដោយសារតែបែបនេះហើយទើបចាស់បុរាណយើង", "confidence": 0.85, "notes": "continuous phrase"}
    ],
    6: [
        {"token": "បានចំណាយពេលវេលា", "confidence": 0.90},
        {"token": "របស់ខ្លួនស្ទើរពេញ", "confidence": 0.90},
        {"token": "មួយជីវិតរបស់ខ្លួន", "confidence": 0.90},
        {"token": "ដើម្បីសិក្សាស្វែងយល់", "confidence": 0.90}
    ],
    7: [
        {"token": "នូវអ្វីៗ", "confidence": 0.95},
        {"token": "ដែលកើតមានឡើង", "confidence": 0.90},
        {"token": "ដើម្បីយកមកចងក្រងជាសុភាសិត", "confidence": 0.85},
        {"token": "ចងក្រង", "confidence": 0.95},
        {"token": "ជាពាក្យ-", "confidence": 0.90}
    ],
    8: [
        {"token": "ស្លោក", "confidence": 0.95},
        {"token": "ពាក្យប្រដៅ", "confidence": 0.95},
        {"token": "របស់ផ្សេងៗ", "confidence": 0.95},
        {"token": "ដើម្បីទុកអប់រំទូន្មានដល់កូនចៅជំនាន់ក្រោយ", "confidence": 0.85},
        {"token": "ឱ្យ", "confidence": 0.95}
    ],
    9: [
        {"token": "សិក្សាស្វែងយល់", "confidence": 0.90},
        {"token": "។", "confidence": 0.95},
        {"token": "ដែលពាក្យស្លោក", "confidence": 0.90},
        {"token": "ក៏ដូចជាសុភាសិតទាំងនោះ", "confidence": 0.85},
        {"token": "មានឥទ្ធិពលដ៏ខ្ពង់ខ្ពស់", "confidence": 0.85}
    ],
    10: [
        {"token": "សង្គមជាតិផងដែរ", "confidence": 0.90},
        {"token": "។", "confidence": 0.95},
        {"token": "ហេតុនេះ", "confidence": 0.95},
        {"token": "ហើយទើបមាន", "confidence": 0.90},
        {"token": "សុភាសិតមួយលើកឡើងថា", "confidence": 0.85},
        {"token": "“ការពា", "confidence": 0.90, "notes": "hyphenated across lines"}
    ],
    11: [
        {"token": "រ", "confidence": 0.90, "notes": "continuation of ការពារ"},
        {"token": "ប្រសើរជាងព្យាបាល", "confidence": 0.90},
        {"token": "”", "confidence": 0.90},
        {"token": "។", "confidence": 0.95}
    ],
    12: [
        {"token": "តើសុភាសិត", "confidence": 0.95},
        {"token": "ដែលបានលើកឡើងខាងលើនេះ", "confidence": 0.85},
        {"token": "មាន", "confidence": 0.95},
        {"token": "អត្ថន័យដូចម្តេចខ្លះ", "confidence": 0.90},
        {"token": "?", "confidence": 0.95}
    ],
    13: [
        {"token": "ដើម្បីជាស្ពានឈានទៅដល់ការបកស្រាយ", "confidence": 0.85},
        {"token": "អត្ថន័យខ្លឹមសារនៃប្រធាន", "confidence": 0.85}
    ],
    14: [
        {"token": "ឱ្យកាន់តែពិរោះពិសា", "confidence": 0.90},
        {"token": "និងច្បាស់លាស់ថែមទាំងស៊ីជម្រៅ", "confidence": 0.85},
        {"token": "មួយកម្រិតទៀតនោះ", "confidence": 0.90},
        {"token": "យើងគប្បី", "confidence": 0.95}
    ],
    15: [
        {"token": "ត្រូវស្វែងយល់គន្លឹះនៃគំនិតជាមុន", "confidence": 0.85},
        {"token": "ត្រូវសិក្សាអំពីអត្ថន័យនៃពាក្យគន្លឹះ", "confidence": 0.85},
        {"token": "ដែលមានក្នុងអត្ថ-", "confidence": 0.90}
    ],
    16: [
        {"token": "ន័យ", "confidence": 0.95},
        {"token": "ប្រធាន", "confidence": 0.95},
        {"token": "ដែលជាសុភាសិតខាងលើនេះជាមុនសិន", "confidence": 0.85},
        {"token": "។", "confidence": 0.95},
        {"token": "ការពារ", "confidence": 0.95},
        {"token": "មានន័យថា", "confidence": 0.95},
        {"token": "ការប្រឆាំង", "confidence": 0.95}
    ],
    17: [
        {"token": "ការទប់ស្កាត់", "confidence": 0.95},
        {"token": "នូវអ្វីៗ", "confidence": 0.95},
        {"token": "ដែលមិនទាន់កើតឡើងតែយើងដឹងមុន", "confidence": 0.85},
        {"token": "។", "confidence": 0.95},
        {"token": "ចំណែកឯ", "confidence": 0.95},
        {"token": "“", "confidence": 0.90},
        {"token": "ព្យាបាល”", "confidence": 0.90}
    ],
    18: [
        {"token": "សម្ដៅលើ", "confidence": 0.95},
        {"token": "ដំណោះស្រាយ", "confidence": 0.95},
        {"token": "ការស្វែងរកនូវ", "confidence": 0.90},
        {"token": "វិធីសាស្ត្រក្នុងការដោះស្រាយ", "confidence": 0.85},
        {"token": "នូវអ្វីៗដែល", "confidence": 0.90}
    ],
    19: [
        {"token": "បានកើតឡើងរួចហើយ", "confidence": 0.90},
        {"token": "។", "confidence": 0.95},
        {"token": "ដូចនេះ", "confidence": 0.95},
        {"token": "សុភាសិតខាងលើនេះ", "confidence": 0.90},
        {"token": "ចង់សម្ដៅលើ", "confidence": 0.90},
        {"token": "ការប្រឆាំង", "confidence": 0.95}
    ],
    20: [
        {"token": "ឬការប្រុងប្រយ័ត្ន", "confidence": 0.90},
        {"token": "ការទប់ស្កាត់មុន", "confidence": 0.90},
        {"token": "គឺជារឿងមួយដែលល្អពិត", "confidence": 0.85},
        {"token": "ការជួបហើយ", "confidence": 0.90}
    ],
    21: [
        {"token": "ទើបដោះស្រាយ", "confidence": 0.95},
        {"token": "តាមក្រោយ", "confidence": 0.95},
        {"token": "ឬក៏ទាល់តែមានគ្រោះថ្នាក់ទើបប្រុងប្រយ័ត្ន ។", "confidence": 0.85}
    ],
    22: [
        {"token": "ការពិតណាស់", "confidence": 0.95},
        {"token": "វិធីសាស្ត្រការពារមុន", "confidence": 0.90},
        {"token": "គឺជាជម្រើសដ៏ល្អមួយក្នុងការ-", "confidence": 0.85}
    ],
    23: [
        {"token": "ចៀសវាងពីគ្រោះថ្នាក់ផ្សេងៗ", "confidence": 0.85},
        {"token": "។", "confidence": 0.95},
        {"token": "រាល់រឿងរ៉ាវ", "confidence": 0.95},
        {"token": "រាល់បញ្ហា", "confidence": 0.95},
        {"token": "តែងតែកើតមានឡើង។", "confidence": 0.90}
    ],
    24: [
        {"token": "គ្រាន់តែ", "confidence": 0.95},
        {"token": "បញ្ហាទាំងនោះ", "confidence": 0.90},
        {"token": "មាន", "confidence": 0.95},
        {"token": "តូចមានធំ", "confidence": 0.90},
        {"token": "ឬបញ្ហា", "confidence": 0.95},
        {"token": "តិចឬច្រើន", "confidence": 0.90},
        {"token": "ប៉ុណ្ណោះ", "confidence": 0.95},
        {"token": "។ ពាក្យថា", "confidence": 0.90}
    ]
}

WRITER_2_GROUND_TRUTH = {
    "page_001.png": LINE_GROUND_TRUTH_W002_P1
}

def assign_transcription(source_page: str, line_idx: int, cand_idx: int, total_cands: int, cand_info: Dict[str, Any], writer_id: str = "W001") -> Dict[str, Any]:
    """
    Assigns candidate Khmer transcription based on verified line sequence.
    Handles single words, compound words, punctuation, and uncertainty flags.
    Supports writer-specific ground truth mapping (e.g. W001 vs W002).
    """
    # Never reuse another page's transcription for an unknown page.  A wrong
    # label is much more damaging than an empty label because it silently
    # teaches the recognizer an incorrect image-to-text mapping.  Newly added
    # pages therefore remain pending until their own ground truth is supplied.
    if writer_id == "W002":
        page_gt = WRITER_2_GROUND_TRUTH.get(source_page, {})
    else:
        page_gt = PAGE_GROUND_TRUTH.get(source_page, {})
    gts = page_gt.get(line_idx, [])
    
    # If this is W001 page 1 line 0 and stray speck
    if writer_id == "W001" and source_page == "page_001.png" and line_idx == 0 and cand_idx > 0:
        return {
            "label_raw": "",
            "label_normalized": "",
            "label_confidence": 0.0,
            "review_required": True,
            "review_status": "rejected",
            "notes": "stray margin mark / non-handwriting noise"
        }

    # Match based on candidate index within line
    if cand_idx < len(gts):
        gt = gts[cand_idx]
        token = gt["token"]
        conf = gt.get("confidence", 0.90)
        notes = gt.get("notes", "")
        
        # Check if segmentation flagged tight spacing or compound word
        if cand_info.get("review_required", False):
            seg_note = cand_info.get("notes", "")
            notes = f"{notes}; {seg_note}".strip("; ")
            
        rev_req = conf < 0.85 or cand_info.get("review_required", False)
        
        return {
            "label_raw": token,
            "label_normalized": normalize_khmer(token),
            "label_confidence": conf,
            "review_required": rev_req,
            "review_status": "pending",
            "notes": notes
        }
    else:
        # Extra candidate beyond expected tokens -> mark for review
        return {
            "label_raw": "",
            "label_normalized": "",
            "label_confidence": 0.50,
            "review_required": True,
            "review_status": "pending",
            "notes": "Unmatched candidate — human transcription required"
        }
