import { defineRailway, github, preserve, project, service } from "railway/iac";

export default defineRailway(() => {
  const preflight = service("preflight", {
    source: github("haomingkoo/preflight", { checkSuites: false }),
    build: { builder: "DOCKERFILE" },
    deploy: {
      healthcheckPath: "/",
      healthcheckTimeout: 60,
      restartPolicyType: "ON_FAILURE",
    },
    replicas: { "asia-southeast1-eqsg3a": 1 },
    domains: ["preflight.kooexperience.com"],
    env: {
      PREFLIGHT_REQUIRE_AUTH: preserve(),
      PREFLIGHT_SECRET_KEY: preserve(),
    },
  });

  return project("serene-victory", {
    resources: [preflight],
  });
});
