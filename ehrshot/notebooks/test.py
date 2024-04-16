import meds, datetime
null = None
patient = {
    'patient_id': 30,
    'events': [{
      "time": datetime.datetime.fromisoformat("2015-12-28 05:17:00"),
      "measurements": [
        {
          "code": "CMS Place of Service/24",
          "text_value": null,
          "numeric_value": null,
          "datetime_value": null,
          "metadata": {
            "care_site_id": "528354",
            "clarity_table": "shc_clarity_adt",
            "end": datetime.datetime.fromisoformat("2015-12-28 05:55:00.000"),
            "note_id": null,
            "table": "visit_detail",
            "unit": null,
            "visit_id": "1233317"
          }
        },
        {
          "code": "Visit/IP",
          "text_value": null,
          "numeric_value": null,
          "datetime_value": null,
          "metadata": {
            "care_site_id": "528534",
            "clarity_table": "shc_pat_enc",
            "end": datetime.datetime.fromisoformat("2015-12-29 17:49:00.000"),
            "note_id": null,
            "table": "visit",
            "unit": null,
            "visit_id": "1233317"
          }
        },
        {
          "code": "SNOMED/443820000",
          "text_value": null,
          "numeric_value": null,
          "datetime_value": null,
          "metadata": {
            "care_site_id": null,
            "clarity_table": "shc_hsp_acct_admit_dx",
            "end": null,
            "note_id": null,
            "table": "condition",
            "unit": null,
            "visit_id": "1233317"
          }
        },
        {
          "code": "SNOMED/93849006",
          "text_value": null,
          "numeric_value": null,
          "datetime_value": null,
          "metadata": {
            "care_site_id": null,
            "clarity_table": "shc_hsp_acct_dx_list",
            "end": null,
            "note_id": null,
            "table": "condition",
            "unit": null,
            "visit_id": "1233317"
          }
        }
      ]
    }]
}
from femr.transforms.stanford import join_consecutive_day_visits

import pprint

pprint.pprint(join_consecutive_day_visits(patient))