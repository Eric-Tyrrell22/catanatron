#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { SelfPlayStack } from "../lib/selfplay-stack";

const app = new cdk.App();

function context(key: string): string {
  const v = app.node.tryGetContext(key);
  if (v === undefined || v === "") {
    throw new Error(`missing context value '${key}' (pass -c ${key}=...)`);
  }
  return String(v);
}

const account = process.env.CDK_DEFAULT_ACCOUNT;
const region = process.env.CDK_DEFAULT_REGION;
if (!account || !region) {
  throw new Error("no AWS account/region resolved: configure the AWS CLI credentials and region");
}

new SelfPlayStack(app, "CatanSelfPlay", {
  env: { account, region },
  instanceTypes: context("instanceTypes").split(",").map((t) => t.trim()),
  maxInstances: Number(context("maxInstances")),
  armInstanceTypes: context("armInstanceTypes").split(",").map((t) => t.trim()),
  armMaxInstances: Number(context("armMaxInstances")),
  spotMaxPriceUsd: Number(context("spotMaxPriceUsd")),
  budgetUsd: Number(context("budgetUsd")),
  alertEmail: context("alertEmail"),
});
