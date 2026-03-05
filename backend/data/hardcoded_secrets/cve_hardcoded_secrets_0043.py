# Source: CVEFixes dataset
# Safety: vulnerable
# Category: hardcoded_secrets

from django.core.exceptions import SuspiciousOperation





class DisallowedModelAdminLookup(SuspiciousOperation):

    """Invalid filter was passed to admin view via URL querystring"""

    pass