# Databricks notebook source
import random, string
import uuid
from datetime import datetime, timedelta
import mmh3
import rstr
from random import randrange

# COMMAND ----------

#define some events and dependencies between the events
advertising_events = {
 
    #eventType          : (weeklyAverageOccurrence, propensityDelta, [(field_to_replace, value)], timeInHoursFromDependent)
    "advertising.clicks": (0.01,                    0.002,            [("advertising/clicks/value", 1.0)], 0.5) , 
    "advertising.impressions": (0.1, 0.001, [("advertising/impressions/value", 1.0)], 0),

    "web.webpagedetails.pageViews": (0.1, 0.005, [("web/webPageDetails/pageViews/value", 1.0)], 0.1),
    "web.webinteraction.linkClicks": (0.05, 0.005, [("web/webInteraction/linkClicks/value", 1.0)], 0.1),
   
    
    "commerce.productViews": (0.05, 0.005, [("commerce/productViews/value", 1.0)], 0.2),
    "commerce.purchases": (0.01, 0.1, [("commerce/purchases/value", 1.0)], 1),
    
    
    "decisioning.propositionDisplay": (0.05, 0.005, [("_experience/decisioning/propositionEventType/display", 1)], 0.1),
    "decisioning.propositionInteract": (0.01, 0.1, [("_experience/decisioning/propositionEventType/interact", 1)], 0.05),
    "decisioning.propositionDismiss": (0.01, -0.2, [("_experience/decisioning/propositionEventType/dismiss", 1)], 0.05),

    
    "directMarketing.emailOpened": (0.2, 0.02, [("directMarketing/opens/value", 1.0)], 24),
    "directMarketing.emailClicked": (0.05, 0.1, [("directMarketing/clicks/value", 1.0)], 0.5),
    "directMarketing.emailSent": (0.5, 0.005, [("directMarketing/sends/value", 1.0)], 0),
    
    "web.formFilledOut": (0.0, 0.0, [("web/webPageDetails/name", "subscriptionForm")], 0),

}

event_dependencies = {
    "advertising.impressions": ["advertising.clicks"],
    "directMarketing.emailSent": ["directMarketing.emailOpened"],
    "directMarketing.emailOpened": ["directMarketing.emailClicked"],
    "directMarketing.emailClicked": ["web.webpagedetails.pageViews"],
    "web.webpagedetails.pageViews": ["web.webinteraction.linkClicks", "commerce.productViews", "decisioning.propositionDisplay"],
    "commerce.productViews": ["commerce.purchases"],
    "decisioning.propositionDisplay": ["decisioning.propositionInteract", "decisioning.propositionDismiss"]
    
}

# COMMAND ----------

# we define some utility functions to be used later

import numpy as np
from datetime import datetime
import math


def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


def create_data_for_n_users(n_users, first_user):

    N_USERS = n_users
    FIRST_USER = first_user

    N_WEEKS = 10
    GLOBAL_START_DATE = datetime.now() - timedelta(weeks=12)
    GLOBAL_END_DATE = GLOBAL_START_DATE + timedelta(weeks=N_WEEKS)

    events = []

    for user in range(N_USERS):
        user_id = FIRST_USER + user
        user_events = []
        base_events = {}
        for event_type in [
            "advertising.impressions",
            "web.webpagedetails.pageViews",
            "directMarketing.emailSent",
        ]:
            n_events = np.random.poisson(advertising_events[event_type][0] * N_WEEKS)
            times = []
            for _ in range(n_events):
                # times.append(random_date(GLOBAL_START_DATE, GLOBAL_END_DATE)
                times.append(
                    random_date(GLOBAL_START_DATE, GLOBAL_END_DATE).isoformat()
                )

            base_events[event_type] = times

        for event_type, dependent_event_types in event_dependencies.items():

            if event_type in base_events:
                # for each originating event
                for event_time in base_events[event_type]:
                    # Look for possible later on events
                    for dependent_event in dependent_event_types:
                        n_events = np.random.poisson(
                            advertising_events[dependent_event][0] * N_WEEKS
                        )
                        times = []
                        for _ in range(n_events):
                            # times.append(event_time + timedelta(hours = np.random.exponential(advertising_events[dependent_event][3])))
                            new_time = datetime.fromisoformat(event_time) + timedelta(
                                hours=np.random.exponential(
                                    advertising_events[dependent_event][3]
                                )
                            )
                            times.append(new_time.isoformat())
                        base_events[dependent_event] = times

        for event_type, times in base_events.items():
            for time in times:
                user_events.append(
                    {"userId": user_id, "eventType": event_type, "timestamp": time}
                )

        user_events = sorted(user_events, key=lambda x: (x["userId"], x["timestamp"]))

        cumulative_probability = 0.001
        subscribed = False
        for event in user_events:
            cumulative_probability = min(
                1.0,
                max(
                    cumulative_probability + advertising_events[event["eventType"]][1],
                    0,
                ),
            )
            event["subscriptionPropensity"] = cumulative_probability
            if (
                subscribed == False
                and "directMarketing" not in event["eventType"]
                and "advertising" not in event["eventType"]
            ):
                subscribed = np.random.binomial(1, cumulative_probability) > 0
                if subscribed:
                    subscriptiontime = (
                        datetime.fromisoformat(event["timestamp"])
                        + timedelta(seconds=60)
                    ).isoformat()
                    # subscriptiontime = event["timestamp"] + timedelta(seconds = 60)
                    user_events.append(
                        {
                            "userId": user_id,
                            "eventType": "web.formFilledOut",
                            "timestamp": subscriptiontime,
                        }
                    )
            event["subscribed"] = subscribed
        user_events = sorted(user_events, key=lambda x: (x["userId"], x["timestamp"]))

        events = events + user_events
    return events

# COMMAND ----------

# utility functions continued
def normalize_ecid(ecid_part):
    ecid_part_str = str(abs(ecid_part))
    if len(ecid_part_str) != 19:
        ecid_part_str = "".join([str(x) for x in range(
            0, 19 - len(ecid_part_str))]) + ecid_part_str
    return ecid_part_str


def get_ecid(email):
    """
    The ECID must be two valid 19 digit longs concatenated together
    """
    ecidpart1, ecidpart2 = mmh3.hash64(email)
    ecid1, ecid2 = (normalize_ecid(ecidpart1), normalize_ecid(ecidpart2))
    return ecid1 + ecid2

# COMMAND ----------

# define the different types of events
def create_email_event(user_id, event_type, timestamp):
    """
    Combines previous methods to create various type of email events
    """

    if event_type == "directMarketing.emailSent":
        directMarketing = {
            "emailDelivered": {"value": 1.0},
            "sends": {"value": 1.0},
            "emailVisitorID": user_id,
            "hashedEmail": "".join(
                random.choices(string.ascii_letters + string.digits, k=10)
            ),
            "messageID": str(uuid.uuid4()),
        }
    elif event_type == "directMarketing.emailOpened":
        directMarketing = {
            "offerOpens": {"value": 1.0},
            "opens": {"value": 1.0},
            "emailVisitorID": user_id,
            "messageID": str(uuid.uuid4()),
        }
    elif event_type == "directMarketing.emailClicked":
        directMarketing = {
            "clicks": {"value": 1.0},
            "offerOpens": {"value": 1.0},
            "emailVisitorID": user_id,
            "messageID": str(uuid.uuid4()),
        }
    return {
        "directMarketing": directMarketing,
        "web": None,
        "_id": str(uuid.uuid4()),
        "eventMergeId": None,
        "eventType": event_type,
        f"_{tenant_id}": {"userid": get_ecid(user_id)},
        "producedBy": "databricks-synthetic",
        "timestamp": timestamp,
    }


def create_web_event(user_id, event_type, timestamp):
    """
    Combines previous methods to creat various type of web events
    """
    url = f"http://www.{''.join(random.choices(string.ascii_letters + string.digits, k=5))}.com"
    ref_url = f"http://www.{''.join(random.choices(string.ascii_letters + string.digits, k=5))}.com"
    name = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    isHomePage = random.choice([True, False])
    server = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    site_section = "".join(random.choices(string.ascii_letters, k=2))
    view_name = "".join(random.choices(string.ascii_letters, k=3))
    region = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    interaction_type = random.choice(["download", "exit", "other"])
    web_referrer = random.choice(
        [
            "internal",
            "external",
            "search_engine",
            "email",
            "social",
            "unknown",
            "usenet",
            "typed_bookmarked",
        ]
    )
    base_web = {
        "webInteraction": {
            "linkClicks": {"value": 0.0},
            "URL": url,
            "name": name,
            "region": region,
            "type": interaction_type,
        },
        "webPageDetails": {
            "pageViews": {"value": 1.0},
            "URL": url,
            "isErrorPage": False,
            # "isHomepage": isHomePage,
            "name": name,
            "server": server,
            "siteSection": site_section,
            "viewName": view_name,
        },
        "webReferrer": {"URL": ref_url, "type": web_referrer},
    }
    if event_type in [
        "advertising.clicks",
        "commerce.purchases",
        "web.webinteraction.linkClicks",
        "web.formFilledOut",
        "decisioning.propositionInteract",
        "decisioning.propositionDismiss",
    ]:
        base_web["webInteraction"]["linkClicks"]["value"] = 1.0

    return {
        "directMarketing": None,
        "web": base_web,
        "_id": str(uuid.uuid4()),
        "eventMergeId": None,
        "eventType": event_type,
        f"_{tenant_id}": {"userid": get_ecid(user_id)},
        "producedBy": "databricks-synthetic",
        "timestamp": timestamp,
    }


def create_xdm_event(user_id, event_type, timestamp):
    """
    The final 'event factory' method that converts an event into an XDM event
    """
    if "directMarketing" in event_type:
        return create_email_event(user_id, event_type, timestamp)
    else:
        return create_web_event(user_id, event_type, timestamp)


